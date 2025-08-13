import { Schema, model } from 'mongoose'

const resultSchema = new Schema(
  {
    filename: { type: String, required: true },
    classification_result: {
      predicted_class: { type: String },
      confidence: { type: String },
      probabilities: {
        type: Map,
        of: String,
      },
    },
    complexity_scores: { type: Object },
    processing_time_seconds: { type: Number },
    version: { type: Number },
    project: { type: String, required: true },
    sequence: { type: String, required: true },
    shot: { type: String, required: true },
    predicted_vfx_hours: { type: Number, default: null },  // <-- Added field here
  },
  {
    timestamps: true,
    collection: 'results',
  }
)

export default model('Result', resultSchema)
