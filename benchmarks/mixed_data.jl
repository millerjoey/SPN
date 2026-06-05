#!/usr/bin/env julia

using CategoricalArrays
using Downloads
using Distributions
using Printf
using Random
using SPN
using StableRNGs
using Statistics
using TypedTables

const DATA_DIR = joinpath(@__DIR__, "data")

const CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
const ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

const CREDIT_NAMES = (:a1, :a2, :a3, :a4, :a5, :a6, :a7, :a8, :a9, :a10, :a11, :a12, :a13, :a14, :a15, :approved)
const CREDIT_NUMERIC = Set([:a2, :a3, :a8, :a11, :a14, :a15])

const ADULT_NAMES = (:age, :workclass, :fnlwgt, :education, :education_num, :marital_status, :occupation, :relationship, :race, :sex, :capital_gain, :capital_loss, :hours_per_week, :native_country, :income)
const ADULT_NUMERIC = Set([:age, :fnlwgt, :education_num, :capital_gain, :capital_loss, :hours_per_week])

struct BenchConfig
    datasets::Vector{String}
    alpha::Float64
    fit_iters::Int
    max_rows::Union{Nothing,Int}
    seed::Int
    train_frac::Float64
    synthetic_missing_rate::Float64
end

function main(args = ARGS)
    cfg = parse_args(args)
    println("dataset,rows,cols,learn_seconds,nodes,sums,products,leaves,params,train_ll,test_ll,fit_seconds,fit_train_ll,fit_test_ll")
    for name in cfg.datasets
        result = run_benchmark(name, cfg)
        print_result(result)
    end
end

function parse_args(args)
    datasets = String[]
    alpha = 0.2
    fit_iters = 0
    max_rows = nothing
    seed = 1
    train_frac = 0.8
    synthetic_missing_rate = 0.05

    for arg in args
        if startswith(arg, "--alpha=")
            alpha = parse(Float64, last(split(arg, "=", limit = 2)))
        elseif startswith(arg, "--fit-iters=")
            fit_iters = parse(Int, last(split(arg, "=", limit = 2)))
        elseif startswith(arg, "--max-rows=")
            max_rows = parse(Int, last(split(arg, "=", limit = 2)))
        elseif startswith(arg, "--seed=")
            seed = parse(Int, last(split(arg, "=", limit = 2)))
        elseif startswith(arg, "--train-frac=")
            train_frac = parse(Float64, last(split(arg, "=", limit = 2)))
        elseif startswith(arg, "--synthetic-missing-rate=")
            synthetic_missing_rate = parse(Float64, last(split(arg, "=", limit = 2)))
        elseif arg in ("-h", "--help")
            print_help()
            exit(0)
        else
            push!(datasets, lowercase(arg))
        end
    end

    isempty(datasets) && (datasets = ["synthetic", "credit", "adult"])
    return BenchConfig(datasets, alpha, fit_iters, max_rows, seed, train_frac, synthetic_missing_rate)
end

function print_help()
    println("""
    Usage:
      julia --project=. benchmarks/mixed_data.jl [synthetic] [credit] [adult] [options]

    Options:
      --alpha=0.2
      --fit-iters=0
      --max-rows=N
      --seed=1
      --train-frac=0.8
      --synthetic-missing-rate=0.05
    """)
end

function run_benchmark(name::String, cfg::BenchConfig)
    D = load_dataset(name, cfg)
    D = limit_rows(D, cfg.max_rows)
    rng = StableRNG(cfg.seed)
    train, test = split_table(D; train_frac = cfg.train_frac, rng = rng)

    learn_seconds = @elapsed spn = learnSPN(train, cfg.alpha)
    θ0, pm = initial_params(spn)
    train_ll = safe_meanlogpdf(spn, train, θ0, pm, name, "train")
    test_ll = isempty_table(test) ? NaN : safe_meanlogpdf(spn, test, θ0, pm, name, "test")
    counts = node_counts(spn.root)

    fit_seconds = cfg.fit_iters > 0 ? NaN : 0.0
    fit_train_ll = NaN
    fit_test_ll = NaN
    if cfg.fit_iters > 0 && isfinite(train_ll)
        fit_seconds = @elapsed begin
            θ, pm, _ = fit_params(spn, train; θ0 = θ0, pm = pm, maxiters = cfg.fit_iters, verbose = false)
        end
        fit_train_ll = safe_meanlogpdf(spn, train, θ, pm, name, "fit_train")
        fit_test_ll = isempty_table(test) ? NaN : safe_meanlogpdf(spn, test, θ, pm, name, "fit_test")
    elseif cfg.fit_iters > 0
        @warn "Skipping parameter fitting because initial train likelihood is not finite" dataset = name train_ll
    end

    return (
        dataset = name,
        rows = nrows(D),
        cols = ncols(D),
        learn_seconds = learn_seconds,
        nodes = counts.nodes,
        sums = counts.sums,
        products = counts.products,
        leaves = counts.leaves,
        params = length(θ0),
        train_ll = train_ll,
        test_ll = test_ll,
        fit_seconds = fit_seconds,
        fit_train_ll = fit_train_ll,
        fit_test_ll = fit_test_ll,
    )
end

function print_result(r)
    @printf(
        "%s,%d,%d,%.4f,%d,%d,%d,%d,%d,%.6f,%.6f,%.4f,%.6f,%.6f\n",
        r.dataset,
        r.rows,
        r.cols,
        r.learn_seconds,
        r.nodes,
        r.sums,
        r.products,
        r.leaves,
        r.params,
        r.train_ll,
        r.test_ll,
        r.fit_seconds,
        r.fit_train_ll,
        r.fit_test_ll,
    )
end

function load_dataset(name::String, cfg::BenchConfig)
    if name == "synthetic"
        return synthetic_mixed_table(1_000; seed = cfg.seed, missing_rate = cfg.synthetic_missing_rate)
    elseif name == "credit"
        return load_credit_approval()
    elseif name == "adult"
        return load_adult()
    end
    error("Unknown dataset $(repr(name)); expected synthetic, credit, or adult.")
end

function fetch_file(url::AbstractString, path::AbstractString)
    mkpath(dirname(path))
    if !isfile(path)
        @info "Downloading benchmark data" url path
        Downloads.download(url, path)
    end
    return path
end

function load_credit_approval()
    path = fetch_file(CREDIT_URL, joinpath(DATA_DIR, "credit_approval", "crx.data"))
    rows = read_csv_rows(path)
    cols = columns_from_rows(rows, CREDIT_NAMES, CREDIT_NUMERIC)
    return Table(NamedTuple{CREDIT_NAMES}(cols))
end

function load_adult()
    path = fetch_file(ADULT_URL, joinpath(DATA_DIR, "adult", "adult.data"))
    rows = read_csv_rows(path)
    cols = columns_from_rows(rows, ADULT_NAMES, ADULT_NUMERIC)
    return Table(NamedTuple{ADULT_NAMES}(cols))
end

function read_csv_rows(path)
    rows = Vector{Vector{String}}()
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        push!(rows, strip.(split(s, ",")))
    end
    return rows
end

function columns_from_rows(rows, names, numeric_names)
    cols = []
    for (j, name) in enumerate(names)
        raw = [row[j] for row in rows]
        if name in numeric_names
            push!(cols, [_parse_numeric(x) for x in raw])
        else
            push!(cols, categorical([_parse_category(x) for x in raw]))
        end
    end
    return cols
end

_is_missing_token(x) = x == "?" || x == "?." || isempty(x)
_parse_numeric(x) = _is_missing_token(x) ? missing : parse(Float64, x)
_parse_category(x) = _is_missing_token(x) ? missing : x

function safe_meanlogpdf(spn, D, θ, pm, dataset, split)
    try
        return meanlogpdf(spn, D, θ, pm)
    catch err
        @warn "Could not score benchmark split" dataset split exception = (err, catch_backtrace())
        return NaN
    end
end

function synthetic_mixed_table(n::Int = 1_000; seed::Int = 1, missing_rate::Real = 0.05)
    rng = StableRNG(seed)
    group = Vector{String}(undef, n)
    signal = Vector{Union{Missing,Float64}}(undef, n)
    positive = Vector{Union{Missing,Float64}}(undef, n)
    count = Vector{Union{Missing,Int}}(undef, n)
    tier = Vector{Union{Missing,String}}(undef, n)
    flag = Vector{Union{Missing,String}}(undef, n)

    for i in 1:n
        z = rand(rng) < 0.55 ? 1 : 2
        group[i] = z == 1 ? "low" : "high"
        signal[i] = z == 1 ? rand(rng, Normal(-1.5, 0.7)) : rand(rng, Normal(2.0, 1.0))
        positive[i] = z == 1 ? rand(rng, Gamma(2.0, 1.0)) : rand(rng, Gamma(7.0, 0.7))
        count[i] = z == 1 ? rand(rng, Poisson(2.0)) : rand(rng, NegativeBinomial(5.0, 0.45))
        tier[i] = z == 1 ? rand(rng, ["A", "A", "B"]) : rand(rng, ["B", "C", "C"])
        flag[i] = (z == 2 && count[i] >= 5) || signal[i] > 1.0 ? "yes" : "no"
    end

    apply_mcar!(rng, signal, missing_rate)
    apply_mcar!(rng, positive, missing_rate)
    apply_mcar!(rng, count, missing_rate)
    apply_mcar!(rng, tier, missing_rate)
    apply_mcar!(rng, flag, missing_rate)

    return Table(
        group = categorical(group),
        signal = signal,
        positive = positive,
        count = count,
        tier = categorical(tier),
        flag = categorical(flag),
    )
end

function apply_mcar!(rng, x, rate)
    for i in eachindex(x)
        rand(rng) < rate && (x[i] = missing)
    end
    return x
end

function split_table(D; train_frac::Real = 0.8, rng = StableRNG(1))
    n = nrows(D)
    n_train = clamp(round(Int, train_frac * n), 1, max(n - 1, 1))
    shuffled = shuffle(rng, collect(1:n))
    train = Set(shuffled[1:n_train])
    ensure_categorical_coverage!(train, D)
    train_idx = sort(collect(train))
    test_idx = setdiff(1:n, train_idx)
    return D[train_idx], D[test_idx]
end

function ensure_categorical_coverage!(train::Set{Int}, D)
    for col in columns(D)
        col isa AbstractCategoricalVector || continue
        for level in levels(col)
            has_level = any(i -> i in train && !ismissing(col[i]) && string(col[i]) == level, eachindex(col))
            has_level && continue
            idx = findfirst(i -> !ismissing(col[i]) && string(col[i]) == level, eachindex(col))
            idx === nothing || push!(train, idx)
        end
    end
    return train
end

function limit_rows(D, max_rows::Nothing)
    return D
end

function limit_rows(D, max_rows::Int)
    return D[1:min(max_rows, nrows(D))]
end

nrows(D) = length(first(columns(D)))
ncols(D) = length(columnnames(D))
isempty_table(D) = nrows(D) == 0

function node_counts(root)
    counts = (nodes = 0, sums = 0, products = 0, leaves = 0)
    return _node_counts(root, counts)
end

function _node_counts(node, counts)
    nodes = counts.nodes + 1
    sums = counts.sums + (node isa SumNode ? 1 : 0)
    products = counts.products + (node isa ProductNode ? 1 : 0)
    leaves = counts.leaves + (node isa Leaf ? 1 : 0)
    next = (nodes = nodes, sums = sums, products = products, leaves = leaves)
    for child in children(node)
        next = _node_counts(child, next)
    end
    return next
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
