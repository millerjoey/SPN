export constructor, scoreSPN

function scoreSPN(SPN::SumProductNetwork, θ, X)
    SPN = recompile(SPN, θ)
    return meanlogpdf(SPN, X)
end

function meanlogpdf(SPN, X)
    n = size(X, 1)
    s = 0.
    for i in 1:n
        s += logpdf(SPN, view(X, i, :))
    end
    return(s/n)
end
