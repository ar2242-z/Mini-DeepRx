function bits = bits(numBits, stream)
%BITS Random information bits from a private random stream.
%   bits = tx.bits(numBits, stream) returns a numBits x 1 column of 0/1.
bits = randi(stream, [0 1], numBits, 1);
end
