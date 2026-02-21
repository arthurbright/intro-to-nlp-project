# stupid backoff TODO do argmax instead of sampling
    BACKOFF_WEIGHT = 0.4
    def sample_next_byte(self, byte_line):
        for gram_size in range(N - 1, -1, -1):
            suffix = byte_line[-gram_size:]  # RAISE 
            if suffix in self.ngram_probs[gram_size]:
                choices = list(self.ngram_probs[gram_size][suffix].keys())
                weights = list(self.ngram_probs[gram_size][suffix].values())
                return random.choices(choices, weights=weights, k=1)[0]
        raise "Should have returned; unigram distribution should always exist"

    def sample_next_char(self, byte_line):
        # assumes byte_line is already padded
        char_bytes = []
        b1 = self.sample_next_byte(byte_line)
        byte_line.append(b1)
        char_bytes.append(b1)
        if(ENCODING == 'utf-8'):
            if b1 < 128 + 64:
                remaining_bytes = 1
            elif b1 < 128 + 64 + 32:
                remaining_bytes = 2
            elif b1 < 128 + 64 + 32 + 16:
                remaining_bytes = 3
            else:
                remaining_bytes = 4
        else:
            raise "UNSPECIFIED ENCODING"
        
        for i in range(remaining_bytes):
            b = self.sample_next_byte(byte_line)
            byte_line.append(b)
            char_bytes.append(b)

        try:
            char = from_bytes(char_bytes)
            return char
        except:
            raise f"Illegal byte sequence: {char_bytes}"