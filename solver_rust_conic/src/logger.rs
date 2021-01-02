pub struct NullLogger;

pub fn null_logger() -> NullLogger
{
    NullLogger{}
}

impl core::fmt::Write for NullLogger
{
    fn write_str(&mut self, _s: &str) -> Result<(), core::fmt::Error>
    {
        Ok(())
    }
}

pub struct IoLogger<'a, W: std::io::Write>(pub &'a mut W);

impl<'a, W: std::io::Write> core::fmt::Write for IoLogger<'a, W>
{
    fn write_str(&mut self, s: &str) -> Result<(), core::fmt::Error>
    {
        if let Ok(_) = self.0.write(s.as_bytes()) {
            Ok(())
        }
        else {
            Err(core::fmt::Error)
        }
    }
}
