pub fn char_to_index(c: char) -> usize {
    if c == '.' {
        return 0;
    }
    // b'a' - ASCII value of 'a' as u8
    (c as u8 - b'a' + 1) as usize
}

pub fn index_to_char(i: usize) -> char {
    if i == 0 {
        return '.';
    }
    (b'a' + i as u8 - 1) as char
}
