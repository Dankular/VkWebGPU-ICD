use std::path::PathBuf;

const VERT_WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>(-0.5, -0.5),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}
"#;

const FRAG_WGSL: &str = r#"
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.0, 1.0);
}
"#;

fn compile_wgsl(source: &str, stage: naga::ShaderStage, entry: &str, out_path: &PathBuf) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("WGSL parse error for {:?}: {}", entry, e));

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation error for {:?}: {}", entry, e));

    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    let pipeline_options = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry.to_string(),
    };

    let words =
        naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options))
            .unwrap_or_else(|e| panic!("SPIR-V write error for {:?}: {}", entry, e));

    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(out_path, &bytes)
        .unwrap_or_else(|e| panic!("Failed to write {:?}: {}", out_path, e));
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out = PathBuf::from(out_dir);

    compile_wgsl(
        VERT_WGSL,
        naga::ShaderStage::Vertex,
        "vs_main",
        &out.join("vert.spv"),
    );
    compile_wgsl(
        FRAG_WGSL,
        naga::ShaderStage::Fragment,
        "fs_main",
        &out.join("frag.spv"),
    );

    println!("cargo:rerun-if-changed=build.rs");
}
