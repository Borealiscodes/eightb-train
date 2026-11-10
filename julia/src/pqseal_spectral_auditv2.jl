module PQSealSpectralAudit

using LinearAlgebra

export pqseal_xor_fold,
       verify_pqseal_manifest_entry,
       unpack_spqt_groupwise,
       audit_pqseal_tensor,
       audit_pqseal_deck_files

"""
    pqseal_xor_fold(data::AbstractVector{UInt8}) -> UInt64

Exact port of `pqseal.py`'s `xor_fold`:
XOR-folds 64-bit little-endian chunks, then XORs remaining tail bytes directly.
"""
function pqseal_xor_fold(data::AbstractVector{UInt8})::UInt64
    acc::UInt64 = 0x0000000000000000
    n = length(data)
    rem_tail = n % 8
    limit = n - rem_tail

    @inbounds for i in 1:8:limit
        word::UInt64 = 0
        for b_idx in 0:7
            word |= UInt64(data[i + b_idx]) << (8 * b_idx)
        end
        acc ⊻= word
    end

    @inbounds for i in (limit + 1):n
        acc ⊻= UInt64(data[i])
    end

    return acc
end

"""
    verify_pqseal_manifest_entry(packed::AbstractVector{UInt8}, manifest_entry::AbstractDict) -> Bool

Verifies a packed SPQT Base-243 tensor payload against its `deck_manifest.json` entry:
- Checks exact packed byte length
- Checks `gate == "0x62"`
- Recomputes the 64-bit `fold` and disjoint `bfp64_pos` / `bfp64_neg` bitmasks
- Recomputes `wire_seal16` over the first 62 bytes (`packed[1:min(62, end)]`)
"""
function verify_pqseal_manifest_entry(
    packed::AbstractVector{UInt8},
    manifest_entry::AbstractDict
)::Bool
    length(packed) == Int(manifest_entry["bytes"]) || return false
    lowercase(String(manifest_entry["gate"])) == "0x62" || return false

    fold = pqseal_xor_fold(packed)
    expected_fold = parse(UInt64, replace(String(manifest_entry["fold"]), "0x" => ""), base = 16)
    fold == expected_fold || return false

    mask::UInt64 = 0x5555555555555555
    pos = fold & mask
    neg = fold & (~mask)
    (pos & neg) == 0x0000000000000000 || return false

    expected_pos = parse(UInt64, replace(String(manifest_entry["bfp64_pos"]), "0x" => ""), base = 16)
    expected_neg = parse(UInt64, replace(String(manifest_entry["bfp64_neg"]), "0x" => ""), base = 16)
    (pos == expected_pos && neg == expected_neg) || return false

    prefix_len = min(62, length(packed))
    seal16 = UInt16(pqseal_xor_fold(view(packed, 1:prefix_len)) & 0xFFFF)
    expected_seal16 = parse(UInt16, replace(String(manifest_entry["wire_seal16"]), "0x" => ""), base = 16)
    return seal16 == expected_seal16
end

"""
    unpack_spqt_groupwise(packed_bytes::AbstractVector{UInt8}, rows::Int, cols::Int,
                          group_alphas_q16::AbstractVector{<:Integer};
                          group_size::Int = 128) -> Matrix{Float64}

Unpacks SPQT Base-243 bytes (5 balanced trits/byte) and dequantizes each group of
`group_size` weights using its matching per-group Q16 Lloyd-Max scale from `.scales.bin`.
"""
function unpack_spqt_groupwise(
    packed_bytes::AbstractVector{UInt8},
    rows::Int,
    cols::Int,
    group_alphas_q16::AbstractVector{<:Integer};
    group_size::Int = 128
)::Matrix{Float64}
    total_weights = rows * cols
    expected_groups = cld(total_weights, group_size)
    length(group_alphas_q16) == expected_groups || error(
        "Scale count mismatch: got $(length(group_alphas_q16)) scales for $expected_groups groups."
    )

    trits = Vector{Int8}(undef, length(packed_bytes) * 5)
    idx = 1
    @inbounds for b in packed_bytes
        b <= 0xF2 || error("Byte 0x$(string(b, base=16)) exceeds Base-243 bound (242).")
        rem_val = Int(b)
        for _ in 1:5
            trits[idx] = Int8((rem_val % 3) - 1)
            rem_val ÷= 3
            idx += 1
        end
    end

    length(trits) >= total_weights || error("Insufficient unpacked trits for $(rows)x$(cols) matrix.")

    flat_weights = Vector{Float64}(undef, total_weights)
    @inbounds for g_idx in 1:expected_groups
        start_i = (g_idx - 1) * group_size + 1
        end_i = min(g_idx * group_size, total_weights)
        scale = Float64(group_alphas_q16[g_idx]) / 65536.0
        for k in start_i:end_i
            flat_weights[k] = Float64(trits[k]) * scale
        end
    end

    W_col = reshape(flat_weights, cols, rows)
    return permutedims(W_col)
end

"""
    audit_pqseal_tensor(packed_bytes::AbstractVector{UInt8}, rows::Int, cols::Int,
                        group_alphas_q16::AbstractVector{<:Integer};
                        group_size::Int = 128,
                        manifest_entry::Union{Nothing, AbstractDict} = nothing,
                        kappa_threshold::Float64 = 4.5,
                        tikhonov_lambda::Float64 = 0.0015) -> NamedTuple

Runs out-of-band spectral SVD verification on a group-wise sealed weight matrix.
Applies the Coq-verified linear Tikhonov shift (sigma + lambda) if rank collapse occurs
and flags `safe_for_inference = false`.
"""
function audit_pqseal_tensor(
    packed_bytes::AbstractVector{UInt8},
    rows::Int,
    cols::Int,
    group_alphas_q16::AbstractVector{<:Integer};
    group_size::Int = 128,
    manifest_entry::Union{Nothing, AbstractDict} = nothing,
    kappa_threshold::Float64 = 4.5,
    tikhonov_lambda::Float64 = 0.0015
)
    if manifest_entry !== nothing && !verify_pqseal_manifest_entry(packed_bytes, manifest_entry)
        return (
            status = :WIRE_SEAL_FAILURE_0x0F,
            safe_for_inference = false,
            kappa_2 = Inf,
            sigma_min = 0.0,
            destroyed_dimensions = min(rows, cols),
            repaired_weights = nothing
        )
    end

    W = unpack_spqt_groupwise(
        packed_bytes, rows, cols, group_alphas_q16; group_size = group_size
    )

    F = svd(W)
    s = F.S
    sigma_max = s[1]
    sigma_min = s[end]
    kappa_2 = sigma_max / max(sigma_min, 1e-15)
    destroyed_dims = count(<(1e-9), s)
    collapsed = (kappa_2 > kappa_threshold) || (destroyed_dims > 0)

    repaired = if collapsed
        s_clamped = s .+ tikhonov_lambda
        F.U * Diagonal(s_clamped) * F.Vt
    else
        W
    end

    return (
        status = collapsed ? :RANK_COLLAPSE_GATED_0x0F : :SPECTRAL_NOMINAL,
        safe_for_inference = !collapsed,
        kappa_2 = round(kappa_2, digits = 4),
        sigma_min = round(sigma_min, digits = 6),
        destroyed_dimensions = destroyed_dims,
        repaired_weights = repaired
    )
end

"""
    audit_pqseal_deck_files(deck_dir::AbstractString, manifest_entry::AbstractDict;
                            kappa_threshold::Float64 = 4.5,
                            tikhonov_lambda::Float64 = 0.0015) -> NamedTuple

Reads `.spqt` and `.scales.bin` directly from `deck_dir` using the metadata stored in
`deck_manifest.json` and runs `audit_pqseal_tensor`.
"""
function audit_pqseal_deck_files(
    deck_dir::AbstractString,
    manifest_entry::AbstractDict;
    kappa_threshold::Float64 = 4.5,
    tikhonov_lambda::Float64 = 0.0015
)
    shape = Int.(manifest_entry["shape"])
    length(shape) == 2 || error("Spectral SVD audit expects a 2D weight matrix, got shape $shape.")
    rows, cols = shape[1], shape[2]
    group_size = Int(get(manifest_entry, "group_size", 128))

    spqt_path = joinpath(deck_dir, String(manifest_entry["spqt_file"]))
    scales_path = joinpath(deck_dir, String(manifest_entry["scales_file"]))

    packed_bytes = read(spqt_path)
    num_scales = div(filesize(scales_path), sizeof(Int64))
    group_alphas = Vector{Int64}(undef, num_scales)
    read!(scales_path, group_alphas)

    return audit_pqseal_tensor(
        packed_bytes,
        rows,
        cols,
        group_alphas;
        group_size = group_size,
        manifest_entry = manifest_entry,
        kappa_threshold = kappa_threshold,
        tikhonov_lambda = tikhonov_lambda
    )
end

end # module PQSealSpectralAudit
