use std::print;
use std::prelude::v1::*;

//

/// Logger using `print!` macro
pub struct PrintLogger;

impl core::fmt::Write for PrintLogger
{
    fn write_str(&mut self, s: &str) -> Result<(), core::fmt::Error>
    {
        print!("{}", s);
        Ok(())
    }
}

//

/// Logger with a `std::io::Write` implementor.
/// 
/// Example of a logger writing to a new file.
/// ```
/// use totsu::logger::IoWriteLogger;
/// use std::fs::File;
/// use std::io::prelude::*;
/// 
/// let mut file = File::create("log.txt").unwrap();
/// let logger = IoWriteLogger(&mut file);
/// ```
pub struct IoWriteLogger<'a, W: std::io::Write>(pub &'a mut W);

impl<'a, W: std::io::Write> core::fmt::Write for IoWriteLogger<'a, W>
{
    fn write_str(&mut self, s: &str) -> Result<(), core::fmt::Error>
    {
        if let Ok(_) = self.0.write_all(s.as_bytes()) {
            if let Ok(_) = self.0.flush() {
                Ok(())
            }
            else {
                Err(core::fmt::Error)
            }
        }
        else {
            Err(core::fmt::Error)
        }
    }
}
