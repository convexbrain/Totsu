pub struct NullLogger;

impl core::fmt::Write for NullLogger
{
    fn write_str(&mut self, _s: &str) -> Result<(), core::fmt::Error>
    {
        Ok(())
    }
}

//

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
