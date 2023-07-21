using Weave

dirs = filter(isdir, joinpath.(@__DIR__, readdir(@__DIR__)))

for dir in dirs
    files = filter(x -> endswith(x, ".jmd"), readdir(dir))
    for file in files
        weave(joinpath(dir, file))
    end
end
