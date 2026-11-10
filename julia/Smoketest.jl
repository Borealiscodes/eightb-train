using .PQSealSpectralAudit

# Simulate a 64x64 adapter matrix (4096 weights -> 820 Base-243 bytes)
mock_deck = rand(UInt8(0):UInt8(242), 820)
fold = pqseal_xor_fold(mock_deck)
mask = 0x5555555555555555
seal16 = UInt16(pqseal_xor_fold(view(mock_deck, 1:62)) & 0xFFFF)

mock_manifest = Dict(
    "bytes" => length(mock_deck),
    "gate" => "0x62",
    "fold" => "0x" * string(fold, base = 16, pad = 16),
    "bfp64_pos" => "0x" * string(fold & mask, base = 16, pad = 16),
    "bfp64_neg" => "0x" * string(fold & ~mask, base = 16, pad = 16),
    "wire_seal16" => "0x" * string(seal16, base = 16, pad = 4)
)

report = audit_pqseal_tensor(mock_deck, 64, 64; manifest_entry = mock_manifest, alpha_q16 = 44564)
println("Status: ", report.status)
println("Safe for Inference: ", report.safe_for_inference)
println("Condition Number (kappa_2): ", report.kappa_2)
println("Destroyed Dimensions: ", report.destroyed_dimensions)
