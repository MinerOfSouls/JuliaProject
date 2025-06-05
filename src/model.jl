module model
using LinearAlgebra
export train_model, use_model

image_size = 28

example_steps = ((l2=0.005, lr=0.1, steps = 200, log_every=10), (l2=0.005, lr=0.01, steps = 300, log_every=25))

function softmax(Z)
    maxiumis = vec(maximum(Z, dims=2))
    Z = exp.(Z .- maxiumis)
    sums = sum(Z, dims=2)
    return Z./sums
end

function cross_entropy(S, T)
    return -sum(T .* log.(S))/size(S, 1)
end

function entropy_gradient(X, S, T)
    return transpose(X) * (S - T)
end

function classify(W, X)
    Z = X * W
    preds = mapslices(argmax, Z; dims=2)
    n = size(Z, 1)
    P = zeros(Float32, n, 10)
    for i in 1:n
        P[i, preds[i]] = 1f0
    end
    return P
end

function accuracy(P, T)
    return 100.0 * sum(P .* T)/size(P, 1)
end

using Printf
function log_data(step, cost, train_acc, val_acc)
    @printf("step: %i\tcost: %.2f\ttrain accuracy: %.2f\tvalidation accuracy %.2f\n", step, cost, train_acc, val_acc)
end

using Plots
function gradient_fit(W0, X, T, X_val, T_val; l2=0.005, lr=1.0, steps = 100, log_every=10)
    n = size(X, 1)
    W = deepcopy(W0)
    t_accuracies = zeros(steps)
    v_accuracies = zeros(steps)
    for i in 1:steps
        S = softmax(X * W)
        cost = cross_entropy(S, T) .+ l2/2*sum(W.*W)
        W = W - lr.*(entropy_gradient(X, S, T)./n + l2.*W)

        predicted_train = classify(W, X)
        train_accuracy = accuracy(predicted_train, T)

        predicted_validation = classify(W, X_val)
        validation_accuracy = accuracy(predicted_validation, T_val)
        t_accuracies[i] = train_accuracy
        v_accuracies[i] = validation_accuracy
        if i == 1 || (i+1)%log_every == 0
            log_data(i, cost, train_accuracy, validation_accuracy)
        end
    end

    p = plot!(1:steps, [t_accuracies, v_accuracies], title="Accuracy evolution")
    display(p)
    return W
end

function onehot_encode(labels)
    new = zeros(size(labels,1), 10)
    for (i, l) in enumerate(labels)
        new[i, l+1] = 1
    end
    return new
end

using MLDatasets, JLD, Random
function train_model(steps)
    s = 60000
    dataset_size = 20000
    # training data
    train = MNIST(split=:train)
    order = randperm(s)
    train_images = reshape(train.features, (image_size*image_size, s))'
    train_images = train_images[order, :]
    train_images = hcat(train_images, vec(ones(s)))
    train_images = train_images[1:dataset_size, :]

    train_labes = train.targets[order]
    train_labes = onehot_encode(train_labes)
    train_labes = train_labes[1:dataset_size, :]
    # test data
    test = MNIST(split=:test)
    test_images = reshape(test.features, (image_size*image_size, 10000))'
    test_images = hcat(test_images, vec(ones(10000)))
    test_labes = onehot_encode(test.targets)

    W = rand(image_size*image_size+1, 10) .* 0.1
    for step in steps
        W = gradient_fit(W, train_images, train_labes, test_images, test_labes; step...)
    end

    @save "trained.jld" W
    return W
end

function use_model(image)
    if isfile("trained.jld")
        @load "trained.jld" 
    else
        print("Model not trained")
        return
    end
    append!(image, [1.0])
    class = classify(W, image')
    return class
end

precompile(train_model, (Tuple, ))

end