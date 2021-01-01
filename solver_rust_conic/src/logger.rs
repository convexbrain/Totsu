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

pub struct IoLogger<'a, W: std::io::Write>
{
    io_writer: &'a mut W
}

pub fn io_logger<'a, W>(io_writer: &'a mut W) -> IoLogger<'a, W>
where W: std::io::Write
{
    IoLogger{ io_writer }
}

impl<'a, W: std::io::Write> core::fmt::Write for IoLogger<'a, W>
{
    fn write_str(&mut self, s: &str) -> Result<(), core::fmt::Error>
    {
        self.io_writer.write(s.as_bytes()).unwrap();
        Ok(())
    }
}
