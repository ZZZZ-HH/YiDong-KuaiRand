# Initialize components
from IPA.ipa import OneRecWithIPA, RewardModel

model = YourOneRecModel()  # Should be encoder-decoder architecture
reward_model = RewardModel(input_dim=768)
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Create IPA trainer
ipa_trainer = OneRecWithIPA(model, reward_model, tokenizer)

# Load user histories
user_histories = load_your_data()  # List of List[str]

# Train with IPA
ipa_trainer.train_ipa(user_histories, num_iterations=3)