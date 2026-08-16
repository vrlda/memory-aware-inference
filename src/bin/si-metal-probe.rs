use std::process::ExitCode;

fn main() -> ExitCode {
    match super_inference::metal::probe() {
        Ok(info) => {
            println!("device={}", info.name);
            println!("registry_id={}", info.registry_id);
            println!(
                "recommended_max_working_set_bytes={}",
                info.recommended_max_working_set_bytes
            );
            println!("max_buffer_bytes={}", info.max_buffer_bytes);
            println!("has_unified_memory={}", info.has_unified_memory);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}
