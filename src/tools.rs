use std::iter::Iterator;

/// For be able to have range iterators
pub struct StepRangeInt {
    start: usize,
    end: usize,
    step: usize,
}

impl StepRangeInt {
    pub fn new(start: usize, end: usize, step: usize) -> StepRangeInt {
        StepRangeInt { start, end, step }
    }
}

impl Iterator for StepRangeInt {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.start < self.end {
            let v = self.start;
            self.start = v + self.step;
            Some(v)
        } else {
            None
        }
    }
}