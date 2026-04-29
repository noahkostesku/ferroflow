use std::collections::VecDeque;
use std::future::Future;
use std::sync::Arc;

use ferroflow_core::{OpId, Tensor};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

/// Unique identifier for a worker node in the distributed scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(pub u32);

/// Thread-safe, cloneable queue of pending operation IDs for a single worker.
///
/// Backed by an `Arc<Mutex<VecDeque<OpId>>>` so multiple handles to the same
/// queue can be held across async tasks.
#[derive(Clone)]
pub struct WorkQueue {
    inner: Arc<Mutex<VecDeque<OpId>>>,
}

impl WorkQueue {
    /// Creates an empty work queue.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Pushes an operation ID to the back of the queue.
    pub async fn push(&self, op_id: OpId) {
        self.inner.lock().await.push_back(op_id);
    }

    /// Pops an operation ID from the front of the queue, or `None` if empty.
    pub async fn pop(&self) -> Option<OpId> {
        self.inner.lock().await.pop_front()
    }

    /// Returns the number of pending operations.
    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
    }

    /// Returns `true` if the queue has no pending operations.
    pub async fn is_empty(&self) -> bool {
        self.inner.lock().await.is_empty()
    }

    /// Pops from the front only if the queue has more than `threshold` items.
    ///
    /// Used by work-stealing: thieves leave small queues alone to avoid
    /// starving the owning worker of its last ops.
    pub async fn steal_if_above(&self, threshold: usize) -> Option<OpId> {
        let mut guard = self.inner.lock().await;
        if guard.len() > threshold {
            guard.pop_front()
        } else {
            None
        }
    }

    /// Atomically drains all items from the queue and returns them in FIFO order.
    ///
    /// Used by the GPU batch path: the worker drains its own queue, selects
    /// which ops can form a batch, and calls [`push_front_bulk`] to re-enqueue
    /// the remainder.
    pub async fn drain_all(&self) -> Vec<OpId> {
        self.inner.lock().await.drain(..).collect()
    }

    /// Pushes `ids` to the front of the queue, preserving their relative order.
    ///
    /// Used to re-enqueue ops that were drained but could not join the current
    /// GPU batch.
    pub async fn push_front_bulk(&self, ids: Vec<OpId>) {
        let mut guard = self.inner.lock().await;
        for id in ids.into_iter().rev() {
            guard.push_front(id);
        }
    }
}

impl Default for WorkQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Messages exchanged between the coordinator and worker nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// A worker requests to steal an operation from another worker's queue.
    StealRequest { from: WorkerId },
    /// Response to a [`Message::StealRequest`]; `op` is `None` when the queue was empty.
    StealResponse { op: Option<OpId> },
    /// A worker has finished computing `op_id` and submits the output tensor.
    OpResult { op_id: OpId, tensor: Tensor },
    /// Instructs the receiving worker to shut down cleanly.
    Shutdown,
}

/// Core interface for a worker participating in the distributed scheduler.
pub trait WorkerTrait {
    /// Executes the operation identified by `op_id` and returns the output tensor.
    ///
    /// # Errors
    /// Returns an error if the operation fails or required inputs are unavailable.
    fn execute_op(&mut self, op_id: OpId) -> impl Future<Output = anyhow::Result<Tensor>> + Send;

    /// Attempts to steal one operation from this worker's local queue.
    /// Returns `None` if the queue is empty.
    fn steal_request(&self) -> impl Future<Output = Option<OpId>> + Send;

    /// Submits a completed operation result upstream (to the coordinator or result store).
    ///
    /// # Errors
    /// Returns an error if the result cannot be delivered.
    fn submit_result(
        &self,
        op_id: OpId,
        tensor: Tensor,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn work_queue_fifo_order() {
        let q = WorkQueue::new();
        assert!(q.is_empty().await);

        q.push(0).await;
        q.push(1).await;
        q.push(2).await;

        assert_eq!(q.len().await, 3);
        assert_eq!(q.pop().await, Some(0));
        assert_eq!(q.pop().await, Some(1));
        assert_eq!(q.pop().await, Some(2));
        assert_eq!(q.pop().await, None);
        assert!(q.is_empty().await);
    }

    #[tokio::test]
    async fn work_queue_clone_shares_state() {
        let q = WorkQueue::new();
        let q2 = q.clone();

        q.push(42).await;
        assert_eq!(q2.pop().await, Some(42));
        assert!(q.is_empty().await);
    }

    #[test]
    fn message_bincode_roundtrip() {
        let messages = vec![
            Message::StealRequest { from: WorkerId(3) },
            Message::StealResponse { op: Some(7) },
            Message::StealResponse { op: None },
            Message::Shutdown,
        ];

        for msg in &messages {
            let encoded = bincode::serialize(msg).expect("serialize");
            let decoded: Message = bincode::deserialize(&encoded).expect("deserialize");
            let re_encoded = bincode::serialize(&decoded).expect("re-serialize");
            assert_eq!(encoded, re_encoded, "roundtrip mismatch for {msg:?}");
        }
    }

    #[test]
    fn message_op_result_roundtrip() {
        let tensor = Tensor::full(&[2, 2], 1.5);
        let msg = Message::OpResult { op_id: 5, tensor };
        let encoded = bincode::serialize(&msg).expect("serialize");
        let decoded: Message = bincode::deserialize(&encoded).expect("deserialize");
        let re_encoded = bincode::serialize(&decoded).expect("re-serialize");
        assert_eq!(encoded, re_encoded);
    }
}
