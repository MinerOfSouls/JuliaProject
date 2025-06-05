using MLDatasets, Random, LinearAlgebra, Printf

# 1) Load a tiny subset (e.g. 1000 train, 1000 test) so it’s fast to iterate
n_train = 1000
n_test  = 1000
train  = MNIST(split=:train)
test   = MNIST(split=:test)

# 2) Flatten → convert to Float32 → normalize to [0,1]
raw_X_train = Float32.(reshape(train.features, (60_000, 28*28))) ./ 255.0
raw_X_test  = Float32.(reshape(test.features,  (10_000, 28*28))) ./ 255.0

# 3) Subsample a random 1 000 so we can quickly debug
perm_tr = randperm(60_000)[1:n_train]
X_train = raw_X_train[perm_tr, :]                # size = (1000, 784)
X_train = hcat(X_train, ones(Float32, n_train))  # append bias column → (1000, 785)

perm_te = randperm(10_000)[1:n_test]
X_test  = raw_X_test[perm_te, :]
X_test  = hcat(X_test,  ones(Float32, n_test))   # (1000, 785)

labels_tr = train.targets[perm_tr]     # 0–9, length=1000
labels_te = test.targets[perm_te]

# 4) One-hot encode
function onehot_encode(labels)
    n = length(labels)
    Y = zeros(Float32, n, 10)
    for i in 1:n
        Y[i, labels[i] + 1] = 1f0
    end
    return Y
end

T_train = onehot_encode(labels_tr)
T_test  = onehot_encode(labels_te)

# 5) Check that X_train is truly in [0.0, 1.0], Float32
println(" ▶ X_train element type: ", eltype(X_train))
println(" ▶ X_train min/max (excluding bias‐column): ",
        minimum(X_train[:, 1:end-1]), "  /  ", maximum(X_train[:, 1:end-1]))

# 6) Define softmax and cross‐entropy
function softmax(Z)
    rowmax = vec(maximum(Z, dims=2))          # (n,)-vector
    Zs = exp.(Z .- rowmax)                    # subtract max per row
    denom = sum(Zs, dims=2)
    return Zs ./ denom                        # each row sums to 1
end

function cross_entropy(S, T)
    # S,T are (n×10)
    return -sum(T .* log.(S)) / size(S, 1)
end

# 7) A correct argmax→one‐hot classifier
function classify_argmax(W, X)
    Z = X * W                    # (n×10) raw logits
    n = size(Z, 1)
    preds = [findmax(Z[i, :])[2] for i in 1:n]
    P = zeros(Float32, n, 10)
    for i in 1:n
        P[i, preds[i]] = 1f0
    end
    return P
end

function accuracy(P, T)
    return 100.0 * sum(P .* T) / size(P, 1)
end

# 8) A tiny gradient‐descent step just to see “is the cost decreasing?”
D = size(X_train, 2)      # 785
C = 10
W = randn(Float32, D, C) .* 0.01  # small random init

l2 = 0.005
lr = 0.1    # if this “blows up” or does nothing, you’ll see it immediately
num_steps = 20

println("\n=== Running a sanity‐check of 20 steps ===")
for step in 1:num_steps
    # a) Forward pass
    S = softmax(X_train * W)                                           # (1000×10)
    cost = cross_entropy(S, T_train) + (l2/2) * sum(W.^2)

    # b) Gradient
    grad = (X_train' * (S .- T_train)) ./ n_train .+ (l2 .* W)
    W .-= lr .* grad

    # c) Compute train & test accuracy
    P_tr = classify_argmax(W, X_train)
    tr_acc = accuracy(P_tr, T_train)

    P_te = classify_argmax(W, X_test)
    te_acc = accuracy(P_te, T_test)

    @printf(" step %2d   cost=%.4f   train_acc=%.2f%%   test_acc=%.2f%%\n",
            step, cost, tr_acc, te_acc)
end
