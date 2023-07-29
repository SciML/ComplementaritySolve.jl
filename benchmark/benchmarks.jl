using Weave, SciMLBenchmarks

function findall_jmd_files(dir=@__DIR__)
    paths = String[]
    for p in readdir(dir)
        if isdir(joinpath(dir, p))
            append!(paths, findall_jmd_files(joinpath(dir, p)))
        elseif endswith(p, ".jmd")
            push!(paths, joinpath(dir, p))
        end
    end
    return paths
end

files = findall_jmd_files()

foreach(f -> SciMLBenchmarks.weave(splitdir(f)..., (:script, :github, :pdf)), files)
