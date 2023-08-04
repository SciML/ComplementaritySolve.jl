function parse_lcp_data(filename)
    file_lines = readlines(filename)
    dim_line = split(file_lines[1], " ")
    m, n = parse.(Int64, dim_line)
    m == n || throw(DimensionMismatch("matrix is not square: dimensions are ($(m),$(n))"))
    M = zeros(Float64, (m, n))
    q = zeros(Float64, m)
    for i in 1:m
        temp_line = split(file_lines[i + 1], " ")
        new_row = parse.(Float64, filter(!=(""), temp_line))
        M[i, :] .= new_row
    end

    temp_line = split(file_lines[end], " ")
    q = parse.(Float64, filter(!=(""), temp_line))

    return (; M=M, q=q)
end
