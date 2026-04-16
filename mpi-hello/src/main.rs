use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    if rank == 0 {
        let msg = vec![42i32];
        world.process_at_rank(1).send(&msg[..]);
        println!("Rank 0: sent {} to rank 1", msg[0]);
    } else if rank == 1 {
        let (msg, _status) = world.process_at_rank(0).receive_vec::<i32>();
        println!("Rank 1: received {} from rank 0", msg[0]);
    }

    println!("Rank {}/{} done", rank, size);
}
