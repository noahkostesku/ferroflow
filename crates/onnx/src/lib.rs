use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use ferroflow_core::{Dag, Op, OpId, OpKind, Tensor};
use tract_onnx::pb::{tensor_shape_proto, type_proto, AttributeProto, GraphProto, ValueInfoProto};
use tract_onnx::prelude::Framework;

// ── public API ────────────────────────────────────────────────────────────────

/// Parse an ONNX model file into a ferroflow [`Dag`].
///
/// Supported ONNX op types: `MatMul`, `Gemm`, `Relu`, `LayerNormalization`,
/// `ReduceMean`, `GlobalAveragePool`, `Softmax`, `BatchNormalization`, `Conv`.
/// All other op types return an error containing the unsupported op name.
///
/// `Gemm` is mapped to [`OpKind::Matmul`]; the optional bias input is dropped
/// and `transB=1` weights are shape-transposed in the accompanying source tensors
/// (see [`load_model`]).
///
/// # Errors
/// Returns an error if the file cannot be read, the proto is malformed, or an
/// unsupported op type is encountered.
pub fn parse_onnx(path: &Path) -> Result<Dag> {
    Ok(load_model(path)?.0)
}

/// Load an ONNX model and return the [`Dag`] together with zero-filled source
/// tensors whose shapes match the ONNX initializers and model inputs.
///
/// Dynamic batch dimensions in the ONNX proto are resolved to 1.
/// `Gemm` weight tensors are stored pre-transposed when `transB=1`.
///
/// # Errors
/// Same as [`parse_onnx`].
pub fn load_model(path: &Path) -> Result<(Dag, HashMap<OpId, Tensor>)> {
    let proto = tract_onnx::onnx()
        .proto_model_for_path(path)
        .with_context(|| format!("loading ONNX model: {}", path.display()))?;

    let graph = proto.graph.as_ref().context("ONNX model has no graph")?;
    build_dag_and_sources(graph)
}

/// Returns a human-readable summary of `dag`: total op count, source/compute
/// split, edge count, a per-type breakdown of compute ops, and a list of any
/// unsupported ONNX op types encountered during parsing.
pub fn dag_summary(dag: &Dag) -> String {
    let n_ops = dag.ops.len();
    let n_edges: usize = dag.ops.iter().map(|op| op.input_ids.len()).sum();
    let n_sources = dag.ops.iter().filter(|op| op.input_ids.is_empty()).count();
    let n_compute = n_ops - n_sources;

    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    let mut unsupported: HashMap<String, usize> = HashMap::new();
    for op in &dag.ops {
        if !op.input_ids.is_empty() {
            if let OpKind::Unsupported { name } = &op.kind {
                *unsupported.entry(name.clone()).or_default() += 1;
            } else {
                *counts.entry(op_kind_label(&op.kind)).or_default() += 1;
            }
        }
    }

    let mut breakdown: Vec<String> = counts
        .iter()
        .map(|(&k, &v)| format!("  {k}: {v}"))
        .collect();
    breakdown.sort();

    let mut lines = format!(
        "{n_ops} ops ({n_sources} sources, {n_compute} compute), {n_edges} edges\n{}",
        breakdown.join("\n")
    );

    if !unsupported.is_empty() {
        let mut u_list: Vec<String> = unsupported
            .iter()
            .map(|(k, &v)| format!("  {k}: {v}"))
            .collect();
        u_list.sort();
        lines.push_str(&format!("\nunsupported (parsed, not executable):\n{}", u_list.join("\n")));
    }

    lines
}

// ── internals ────────────────────────────────────────────────────────────────

fn op_kind_label(kind: &OpKind) -> &'static str {
    match kind {
        OpKind::Matmul { .. } => "matmul",
        OpKind::Relu { .. } => "relu",
        OpKind::LayerNorm { .. } => "layer_norm",
        OpKind::Reduce { .. } => "reduce",
        OpKind::Softmax { .. } => "softmax",
        OpKind::BatchNorm { .. } => "batch_norm",
        OpKind::Conv2d { .. } => "conv2d",
        OpKind::Slow { .. } => "slow",
        OpKind::Unsupported { .. } => "unsupported",
    }
}

fn build_dag_and_sources(graph: &GraphProto) -> Result<(Dag, HashMap<OpId, Tensor>)> {
    let shape_map = build_shape_map(graph);

    // Collect which initializer names need transposition (Gemm transB=1 weights).
    let mut transpose_needed: HashSet<String> = HashSet::new();
    for node in &graph.node {
        if node.op_type == "Gemm" && get_int_attr(&node.attribute, "transB").unwrap_or(0) == 1 {
            if let Some(w) = node.input.get(1).filter(|s| !s.is_empty()) {
                transpose_needed.insert(w.clone());
            }
        }
    }

    // Initializer names set — used to skip them if they also appear in graph.input
    // (ONNX opset < 9 includes initializers in the input list).
    let init_names: HashSet<&str> = graph.initializer.iter().map(|t| t.name.as_str()).collect();

    let mut tensor_to_id: HashMap<String, OpId> = HashMap::new();
    let mut ops: Vec<Op> = Vec::new();
    let mut sources: HashMap<OpId, Tensor> = HashMap::new();

    // ── model inputs ──────────────────────────────────────────────────────────
    for vi in &graph.input {
        if init_names.contains(vi.name.as_str()) {
            continue;
        }
        let id = ops.len();
        tensor_to_id.insert(vi.name.clone(), id);
        let shape = shape_from_value_info(vi).unwrap_or_else(|| vec![1]);
        let len = shape.iter().product::<usize>().max(1);
        sources.insert(id, Tensor::zeros(&shape));
        ops.push(Op::new(id, OpKind::Relu { len }, vec![], shape));
    }

    // ── initializers ─────────────────────────────────────────────────────────
    for init in &graph.initializer {
        let id = ops.len();
        tensor_to_id.insert(init.name.clone(), id);

        let mut shape: Vec<usize> = init.dims.iter().map(|&d| d.max(1) as usize).collect();
        if shape.is_empty() {
            shape = vec![1];
        }
        if transpose_needed.contains(&init.name) && shape.len() == 2 {
            shape.swap(0, 1);
        }

        let len = shape.iter().product::<usize>().max(1);
        sources.insert(id, Tensor::zeros(&shape));
        ops.push(Op::new(id, OpKind::Relu { len }, vec![], shape));
    }

    // ── computation nodes ─────────────────────────────────────────────────────
    for node in &graph.node {
        let id = ops.len();
        let max_inputs = op_max_inputs(&node.op_type);

        let input_ids: Vec<OpId> = node
            .input
            .iter()
            .take(max_inputs)
            .filter(|name| !name.is_empty())
            .map(|name| {
                tensor_to_id.get(name).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "node '{}' ({}): unknown input tensor '{}'",
                        node.name,
                        node.op_type,
                        name
                    )
                })
            })
            .collect::<Result<_>>()?;

        let out_name = node.output.first().map(String::as_str).unwrap_or("");
        let output_shape = shape_map.get(out_name).cloned().unwrap_or_else(|| vec![1]);

        let kind = map_op_kind(&node.op_type, &output_shape, &node.attribute)?;

        for out in &node.output {
            if !out.is_empty() {
                tensor_to_id.insert(out.clone(), id);
            }
        }

        ops.push(Op::new(id, kind, input_ids, output_shape));
    }

    let dag = Dag::new(ops).context("constructing DAG from ONNX graph")?;
    Ok((dag, sources))
}

/// Maximum number of inputs the mapped OpKind consumes.
/// `usize::MAX` means "all inputs".
fn op_max_inputs(op_type: &str) -> usize {
    match op_type {
        // Gemm: A · B + C — drop C (bias) since OpKind::Matmul takes exactly 2.
        "Gemm" => 2,
        // LayerNormalization: (input, scale, bias) — drop scale/bias.
        "LayerNormalization" => 1,
        // Conv: (input, weight[, bias]) — drop bias for now.
        "Conv" => 2,
        _ => usize::MAX,
    }
}

fn map_op_kind(op_type: &str, output_shape: &[usize], attrs: &[AttributeProto]) -> Result<OpKind> {
    let len = output_shape.iter().product::<usize>().max(1);
    match op_type {
        "MatMul" => {
            let (m, n) = last_two_dims(output_shape, len);
            Ok(OpKind::Matmul { m, n, k: m })
        }
        "Gemm" => {
            let (m, n) = last_two_dims(output_shape, len);
            // k ≈ n (weight was (out, in), transposed to (in, out); k = out = n)
            Ok(OpKind::Matmul { m, n, k: n })
        }
        "Relu" => Ok(OpKind::Relu { len }),
        "LayerNormalization" => Ok(OpKind::LayerNorm { len }),
        "ReduceMean" | "GlobalAveragePool" => {
            let axis = get_int_attr(attrs, "axes")
                .or_else(|| get_int_attr(attrs, "axis"))
                .unwrap_or(0) as usize;
            Ok(OpKind::Reduce { axis, len })
        }
        "Softmax" => Ok(OpKind::Softmax { len }),
        "BatchNormalization" => {
            let epsilon = get_float_attr(attrs, "epsilon").unwrap_or(1e-5);
            Ok(OpKind::BatchNorm { epsilon })
        }
        "Conv" => {
            let kernel_shape = get_int_list_attr(attrs, "kernel_shape");
            let kernel_size = kernel_shape.first().copied().unwrap_or(1) as usize;
            let strides = get_int_list_attr(attrs, "strides");
            let stride = strides.first().copied().unwrap_or(1) as usize;
            let pads = get_int_list_attr(attrs, "pads");
            let padding = pads.first().copied().unwrap_or(0) as usize;
            let dilations = get_int_list_attr(attrs, "dilations");
            if dilations.iter().any(|&d| d != 1) {
                bail!("Conv with dilations != 1 is not supported");
            }
            Ok(OpKind::Conv2d { kernel_size, stride, padding })
        }
        other => Ok(OpKind::Unsupported { name: other.to_string() }),
    }
}

fn last_two_dims(shape: &[usize], fallback_len: usize) -> (usize, usize) {
    if shape.len() >= 2 {
        (shape[shape.len() - 2], shape[shape.len() - 1])
    } else {
        (1, fallback_len)
    }
}

fn build_shape_map(graph: &GraphProto) -> HashMap<String, Vec<usize>> {
    let mut map = HashMap::new();
    for vi in graph
        .value_info
        .iter()
        .chain(graph.input.iter())
        .chain(graph.output.iter())
    {
        if let Some(shape) = shape_from_value_info(vi) {
            map.insert(vi.name.clone(), shape);
        }
    }
    map
}

fn shape_from_value_info(vi: &ValueInfoProto) -> Option<Vec<usize>> {
    let tp = vi.r#type.as_ref()?;
    match tp.value.as_ref()? {
        type_proto::Value::TensorType(tt) => {
            let shape = tt.shape.as_ref()?;
            Some(
                shape
                    .dim
                    .iter()
                    .map(|d| match d.value.as_ref() {
                        Some(tensor_shape_proto::dimension::Value::DimValue(v)) if *v > 0 => {
                            *v as usize
                        }
                        _ => 1, // dynamic or unset — use 1
                    })
                    .collect(),
            )
        }
    }
}

fn get_int_attr(attrs: &[AttributeProto], name: &str) -> Option<i64> {
    attrs.iter().find(|a| a.name == name).map(|a| a.i)
}

fn get_float_attr(attrs: &[AttributeProto], name: &str) -> Option<f32> {
    attrs.iter().find(|a| a.name == name).map(|a| a.f)
}

fn get_int_list_attr(attrs: &[AttributeProto], name: &str) -> Vec<i64> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
        .unwrap_or_default()
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ferroflow_core::{Op, OpKind};

    /// Manually-encoded ONNX proto for: input X (shape [1,4]) → Relu → output Y (shape [1,4]).
    ///
    /// Encoding follows proto3 wire format directly — no prost/protobuf dependency needed
    /// in test code. Byte layout is verified in a comment block below.
    ///
    /// ModelProto  { ir_version=7, graph: GraphProto {
    ///   node:       NodeProto   { input="X", output="Y", op_type="Relu" }
    ///   input:      ValueInfoProto("X", float32, [1,4])
    ///   output:     ValueInfoProto("Y", float32, [1,4])
    ///   value_info: ValueInfoProto("Y", float32, [1,4])
    /// }}
    #[rustfmt::skip]
    const RELU_ONNX: &[u8] = &[
        // ModelProto
        0x08, 0x07,         // ir_version = 7  (field 1, varint)
        0x3A, 0x4D,         // graph (field 7, len=77)
        //   GraphProto
        //   node (field 1, len=12): NodeProto { input="X", output="Y", op_type="Relu" }
        0x0A, 0x0C,
          0x0A, 0x01, 0x58,                         // input  "X"
          0x12, 0x01, 0x59,                         // output "Y"
          0x22, 0x04, 0x52, 0x65, 0x6C, 0x75,       // op_type "Relu"
        //   input (field 11, len=19): ValueInfoProto("X", float32, [1,4])
        0x5A, 0x13,
          0x0A, 0x01, 0x58,                         // name "X"
          0x12, 0x0E,                               // type (len=14)
            0x0A, 0x0C,                             //   tensor_type (len=12)
              0x08, 0x01,                           //     elem_type=1 (float32)
              0x12, 0x08,                           //     shape (len=8)
                0x0A, 0x02, 0x08, 0x01,             //       dim value=1
                0x0A, 0x02, 0x08, 0x04,             //       dim value=4
        //   output (field 12, len=19): ValueInfoProto("Y", float32, [1,4])
        0x62, 0x13,
          0x0A, 0x01, 0x59,
          0x12, 0x0E,
            0x0A, 0x0C,
              0x08, 0x01,
              0x12, 0x08,
                0x0A, 0x02, 0x08, 0x01,
                0x0A, 0x02, 0x08, 0x04,
        //   value_info (field 13, len=19): ValueInfoProto("Y", float32, [1,4])
        0x6A, 0x13,
          0x0A, 0x01, 0x59,
          0x12, 0x0E,
            0x0A, 0x0C,
              0x08, 0x01,
              0x12, 0x08,
                0x0A, 0x02, 0x08, 0x01,
                0x0A, 0x02, 0x08, 0x04,
    ];

    fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn parse_single_relu_op_count_and_edges() {
        let path = write_tmp("ferroflow_relu.onnx", RELU_ONNX);
        let dag = parse_onnx(&path).unwrap();
        // 1 source (input X) + 1 compute (Relu)
        assert_eq!(dag.ops.len(), 2, "op count");
        let edges: usize = dag.ops.iter().map(|op| op.input_ids.len()).sum();
        assert_eq!(edges, 1, "edge count");
        assert!(matches!(dag.ops[1].kind, OpKind::Relu { .. }));
        assert_eq!(dag.ops[1].input_ids, vec![0]);
    }

    #[test]
    fn load_model_provides_source_tensor() {
        let path = write_tmp("ferroflow_relu2.onnx", RELU_ONNX);
        let (dag, sources) = load_model(&path).unwrap();
        assert!(
            sources.contains_key(&0),
            "source tensor for input X must exist"
        );
        assert_eq!(sources[&0].shape(), &[1, 4]);
        assert_eq!(dag.ops.len(), 2);
    }

    #[test]
    fn dag_summary_format() {
        let ops = vec![
            Op::new(0, OpKind::Relu { len: 4 }, vec![], vec![4]),
            Op::new(1, OpKind::Relu { len: 4 }, vec![0], vec![4]),
            Op::new(
                2,
                OpKind::Matmul { m: 2, n: 2, k: 2 },
                vec![0, 1],
                vec![2, 2],
            ),
        ];
        let dag = Dag::new(ops).unwrap();
        let s = dag_summary(&dag);
        assert!(s.contains("3 ops"), "got: {s}");
        assert!(s.contains("1 sources"), "got: {s}");
        assert!(s.contains("2 compute"), "got: {s}");
        assert!(s.contains("3 edges"), "got: {s}");
        assert!(s.contains("matmul: 1"), "got: {s}");
        assert!(s.contains("relu: 1"), "got: {s}");
    }

    #[test]
    fn unsupported_op_parses_and_appears_in_summary() {
        // Reuse RELU_ONNX but patch op_type bytes to "MaxP" (4 bytes, same length as Relu).
        let mut bytes = RELU_ONNX.to_vec();
        let relu = b"Relu";
        let maxp = b"MaxP";
        if let Some(pos) = bytes.windows(4).position(|w| w == relu) {
            bytes[pos..pos + 4].copy_from_slice(maxp);
        }
        let path = write_tmp("ferroflow_maxp.onnx", &bytes);
        let dag = parse_onnx(&path).expect("parse must succeed for unknown op");
        let summary = dag_summary(&dag);
        assert!(summary.contains("MaxP"), "summary should name the unsupported op: {summary}");
        assert!(
            matches!(dag.ops[1].kind, OpKind::Unsupported { .. }),
            "op kind should be Unsupported"
        );
    }
}
