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

    % The two target planes hold the first and second QPSK bit.
    bitLabels = zeros(size(X,1),size(X,2),2,'single');
    for b = 1:2
        plane = zeros(size(X),'single');
        plane(meta.DataIndices) = single(meta.Bits(b:2:end));
        bitLabels(:,:,b) = plane;
    end
    bitMask = repmat(meta.DataMask,1,1,2);
    assert(nnz(bitMask)==numel(meta.Bits),'Target mask and bit count disagree.');
end
