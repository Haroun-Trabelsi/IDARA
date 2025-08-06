// models/Feedback.ts
import { type Document, model, Schema } from 'mongoose';

interface IFeedback extends Document {
  accountId: Schema.Types.ObjectId; // Référence à l'utilisateur (Account)
  rating: number; // Note (1-5)
  feedbackText: string; // Texte du feedback
  featureSuggestions: string[]; // Suggestions de fonctionnalités
  createdAt: Date;
  updatedAt: Date;
}

const instance = new Schema<IFeedback>(
  {
    accountId: {
      type: Schema.Types.ObjectId,
      ref: 'Account',
      required: true,
    },
    rating: {
      type: Number,
      required: true,
      min: 1,
      max: 5,
    },
    feedbackText: {
      type: String,
      required: false,
    },
    featureSuggestions: {
      type: [String],
      required: false,
    },
  },
  {
    timestamps: true,
  }
);

const modelName = 'Feedback';

export default model<IFeedback>(modelName, instance);