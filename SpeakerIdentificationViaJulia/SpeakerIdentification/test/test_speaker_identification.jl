using Test
using SpeakerIdentification

@testset "Speaker Identification Tests" begin
    test_audio = "sample.wav"
    test_features = extract_features(test_audio)
    @test length(test_features) > 0

    train_files = ["speaker1.wav", "speaker2.wav"]
    labels = ["Speaker1", "Speaker2"]
    model = train_speaker_model(train_files, labels)

    predicted_speaker = predict_speaker(model, "speaker1.wav")
    @test predicted_speaker in labels
end
