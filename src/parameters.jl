function find_CUDA_batch_size(instance::MDVRP_Instance)
    CustDepRatio = instance.N/instance.D
    if CustDepRatio < 100
        mini_batch_len = 512            ### based on the CUDA memory 8 GB
    elseif CustDepRatio < 200
        mini_batch_len = 256
    elseif CustDepRatio < 300
        mini_batch_len = 128
    elseif CustDepRatio < 400
        mini_batch_len = 64
    else
        mini_batch_len = 32
    end
    return mini_batch_len
end

function find_HGS_time(instance::MDVRP_Instance)  
    CustDepRatio = instance.N/instance.D
    if CustDepRatio < 100
        hgs_time = 0.5
    elseif CustDepRatio < 200
        hgs_time = 1.0
    else
        hgs_time = 2.0
    end
    return hgs_time
end