
use std::env;

pub fn getenv(var: &str, default: &str) -> String {
	return match env::var(var) {
		Ok(val) => val,
		Err(_err) => default.to_string()
	};
}

#[link(name="bb", kind="static")]
extern "C" {
pub fn _BB();
}

#[macro_export]
macro_rules! BB {
	() => {
		use readies_wd40::_BB;
		use readies_wd40::getenv;
		if getenv("BB", "") == "1" {
			unsafe { _BB(); } 
		}
	};
}
