#![allow(dead_code)]

pub trait Array {
    fn shape(&self) -> &[usize];
}
