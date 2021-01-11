pub struct NullLogger;

impl core::fmt::Write for NullLogger
{
    fn write_str(&mut self, _s: &str) -> Result<(), core::fmt::Error>
    {
        Ok(())
    }
}
