using WAV, MFCC, NearestNeighbors, Statistics, Test

function extract_features(audio_file)
    audio, fs = wavread(audio_file)
    # Compute pitch using zero-crossing rate
    pitch = compute_pitch(audio, fs)
    # Compute MFCCs
    mfccs = mfcc(audio, fs)
    # Concatenate features
    return vcat(mean(pitch), mean(mfccs, dims=2)[:])
end

function train_speaker_model(train_files, labels)
    features = [extract_features(f) for f in train_files]
    feature_matrix = hcat(features...)
    mean_vec = mean(feature_matrix, dims=2)
    std_vec = std(feature_matrix, dims=2)

    # Prevent division by zero
    std_vec[std_vec .== 0] .= 1

    norm_features = (feature_matrix .- mean_vec) ./ std_vec
    knn = KNN(norm_features, labels)

    return knn, mean_vec, std_vec
end

function predict_speaker(knn, mean_vec, std_vec, test_file)
    feature = extract_features(test_file)
    norm_feature = (feature .- mean_vec) ./ std_vec
    return knn_predict(knn, norm_feature)
end


