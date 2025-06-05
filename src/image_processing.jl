module ImageProcessing
using Images, FileIO
function image_to_vector(img)
    img = map(c -> alpha(c) == 0 ? RGBA(0, 0, 0, 1) : c, img)
    matrix = real.(imresize(Gray.(img), (28, 28)))
    return float(vec(matrix))
end

function vector_to_image(vector)
    n = size(vector)[1]
    matrix = reshape(vector[1:n-1], (28, 28))
    image = Gray.(matrix)
    return image
end


using Plots
function display_model(model)
    for i in axes(model, 2)
        im = vector_to_image(vec(model[:, i])) .* 5
        a = histogram2d(im)
        display(a)
    end

end

end