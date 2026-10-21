function [Z, bitLabels, bitMask] = deeprx_sample(Y,X,meta)
%DEEPRX_SAMPLE Shared input/label packing for training and testing.
% Z is (frequency,time,features). For Nr=2 its ten channels are:
% real(Y1,Y2,Xp,Hraw1,Hraw2), then imag(Y1,Y2,Xp,Hraw1,Hraw2).
% ONLY pilot entries of X enter Z. True H and noise variance are never used.
    Xp = zeros(size(X),'like',X);
    Xp(meta.PilotMask) = X(meta.PilotMask);
    Hraw = Y.*conj(Xp); % Raw LS, zero outside pilots; do not interpolate.
    Zc = cat(3,Y,Xp,Hraw);
    Z = single(cat(3,real(Zc),imag(Zc)));
    assert(all(isfinite(Z(:))),'Network inputs must be finite.');

    % One target plane per bit of the symbol (4 for 16QAM). As in the paper,
    % planes 1-2 are the quadrant bits and planes 3-4 refine within it.
    q = meta.BitsPerSymbol;
    bitLabels = zeros(size(X,1),size(X,2),q,'single');
    for b = 1:q
        plane = zeros(size(X),'single');
        plane(meta.DataIndices) = single(meta.Bits(b:q:end));
        bitLabels(:,:,b) = plane;
    end
    bitMask = repmat(meta.DataMask,1,1,q);
    assert(nnz(bitMask)==numel(meta.Bits),'Target mask and bit count disagree.');
end
