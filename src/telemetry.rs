//! Lightweight process and backend allocation sampling.

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceSnapshot {
    pub rss_bytes: u64,
    pub minor_page_faults: u64,
    pub major_page_faults: u64,
    pub voluntary_context_switches: u64,
    pub involuntary_context_switches: u64,
}

impl ResourceSnapshot {
    pub fn delta_since(self, before: Self) -> Self {
        Self {
            rss_bytes: self.rss_bytes,
            minor_page_faults: self
                .minor_page_faults
                .saturating_sub(before.minor_page_faults),
            major_page_faults: self
                .major_page_faults
                .saturating_sub(before.major_page_faults),
            voluntary_context_switches: self
                .voluntary_context_switches
                .saturating_sub(before.voluntary_context_switches),
            involuntary_context_switches: self
                .involuntary_context_switches
                .saturating_sub(before.involuntary_context_switches),
        }
    }
}

#[cfg(target_os = "macos")]
pub fn sample_resources() -> Result<ResourceSnapshot, String> {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if status != 0 {
        return Err(format!("getrusage failed with status {status}"));
    }
    Ok(ResourceSnapshot {
        rss_bytes: current_rss_bytes()?,
        minor_page_faults: usage.ru_minflt as u64,
        major_page_faults: usage.ru_majflt as u64,
        voluntary_context_switches: usage.ru_nvcsw as u64,
        involuntary_context_switches: usage.ru_nivcsw as u64,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn sample_resources() -> Result<ResourceSnapshot, String> {
    Err("process resource sampler requires macOS".into())
}

#[cfg(target_os = "macos")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ProcessMemorySampler {
    peak_rss_bytes: u64,
}

#[cfg(target_os = "macos")]
impl ProcessMemorySampler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sample(&mut self) -> Result<u64, String> {
        let current = current_rss_bytes()?;
        self.peak_rss_bytes = self.peak_rss_bytes.max(current);
        Ok(current)
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.peak_rss_bytes = current_rss_bytes()?;
        Ok(())
    }

    pub fn peak_rss_bytes(&self) -> u64 {
        self.peak_rss_bytes
    }
}

#[cfg(target_os = "macos")]
pub fn current_rss_bytes() -> Result<u64, String> {
    // SAFETY: task_info fills the fully initialized mach_task_basic_info
    // structure and the count is the size of that structure in natural_t
    // units, as required by the Mach API.
    let mut info: libc::mach_task_basic_info = unsafe { std::mem::zeroed() };
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    #[allow(deprecated)]
    let task = unsafe { libc::mach_task_self() };
    let status = unsafe {
        libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut libc::mach_task_basic_info as libc::task_info_t,
            &mut count,
        )
    };
    if status != libc::KERN_SUCCESS {
        return Err(format!("task_info failed with Mach status {status}"));
    }
    Ok(info.resident_size as u64)
}

#[cfg(not(target_os = "macos"))]
pub fn current_rss_bytes() -> Result<u64, String> {
    Err("process RSS sampler requires macOS".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_delta_saturates_counters_and_keeps_rss() {
        let before = ResourceSnapshot {
            rss_bytes: 100,
            minor_page_faults: 12,
            major_page_faults: 4,
            voluntary_context_switches: 9,
            involuntary_context_switches: 7,
        };
        let after = ResourceSnapshot {
            rss_bytes: 80,
            minor_page_faults: 18,
            major_page_faults: 3,
            voluntary_context_switches: 14,
            involuntary_context_switches: 6,
        };

        let delta = after.delta_since(before);

        assert_eq!(delta.rss_bytes, 80);
        assert_eq!(delta.minor_page_faults, 6);
        assert_eq!(delta.major_page_faults, 0);
        assert_eq!(delta.voluntary_context_switches, 5);
        assert_eq!(delta.involuntary_context_switches, 0);
    }
}
