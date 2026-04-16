// placeholder — filled in during Step 4
use criterion::{criterion_group, criterion_main, Criterion};
fn placeholder(c: &mut Criterion) { c.bench_function("placeholder", |b| b.iter(|| 0u64)); }
criterion_group!(benches, placeholder);
criterion_main!(benches);
