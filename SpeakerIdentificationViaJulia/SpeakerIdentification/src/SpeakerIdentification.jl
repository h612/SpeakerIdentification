module SpeakerIdentification

using WAV, MFCC, NearestNeighbors, Statistics

export train_speaker_model, predict_speaker,extract_features  

struct SpeakerModel
    tree::KDTree
    labels::Vector{String}
    mean::Vector{Float64}
    std::Vector{Float64}
end

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
    norm_features = (feature_matrix .- mean_vec) ./ std_vec
    tree = KDTree(norm_features)
    return SpeakerModel(tree, labels, mean_vec, std_vec)
end

function predict_speaker(model::SpeakerModel, test_file)
    feature = extract_features(test_file)
    norm_feature = (feature .- model.mean) ./ model.std
    idx, _ = knn(model.tree, norm_feature, 1)  # Find the nearest neighbor
    return model.labels[idx[1]]  # Return the corresponding label
end

end # module SpeakerIdentification
