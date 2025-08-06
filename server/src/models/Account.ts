import { type Document, model, Schema } from 'mongoose';
import { type Account } from '../@types';

interface I extends Document, Account {
  _id: string;
  createdAt: Date;
  updatedAt: Date;
  organizationName: string;
  invitedBy?: string; // Référence à l'ID de l'utilisateur qui a invité
  canInvite: boolean; // Indique si l'utilisateur peut inviter d'autres collaborateurs
  status?: 'pending' | 'accepted' | 'expired'; // Statut de l'invitation
  invitedDate?: Date; // Date d'invitation
  verificationToken?: string; // Token pour la vérification
  verificationTokenExpires?: Date; // Date d'expiration du token
  mustCompleteProfile: boolean; // Nouveau champ
  receiveUpdates: boolean;
  organizationSize: number; // Nouveau champ, initialisé à 5
  feedbackText?: string; // Champ pour le texte du feedback (textarea)
  featureSuggestions?: string[]; // Tableau pour les suggestions de fonctionnalités
  rating?: number; // Champ pour la note

}

const instance = new Schema<I>(
  {
    name: {
      type: String,
      required: false,
    },
    surname: {
      type: String,
      required: false,
    },
    email: {
      type: String,
      required: true,
      lowercase: true,
      unique: true,
    },
    password: {
      type: String,
      required: true,
    },
    role: {
      type: String,
      required: true,
      enum: ['user', 'admin'],
      default: 'user',
    },
    isVerified: {
      type: Boolean,
      required: true,
      default: false,
    },
    verificationCode: {
      type: String,
      required: false,
    },
    verificationCodeExpires: {
      type: Date,
      required: false,
    },
    mfaSecret: { 
      type: String,
    },
    mfaEnabled: { 
      type: Boolean,
      default: false,
    },
    organizationName: {
      type: String,
      required: true, // Chaque compte doit appartenir à une organisation
    },
    invitedBy: {
      type: Schema.Types.ObjectId,
      ref: 'Account', // Référence à un autre utilisateur Account
      required: false, // Seulement pour les invités
    },
    canInvite: {
      type: Boolean,
      default: true, // Par défaut, seul l'admin initial peut inviter
    },
    status: {
      type: String,
      enum: ['pending', 'accepted', 'expired', 'Administrator'],
      default: 'pending', // Par défaut pour les invitations
    },
    invitedDate: {
      type: Date,
      required: false,
    },
    verificationToken: {
      type: String,
      required: false,
    },
    verificationTokenExpires: {
      type: Date,
      required: false,
    },
    mustCompleteProfile: { type: Boolean, default: false }, // Nouveau champ

    receiveUpdates: { type: Boolean, default: false },

    organizationSize: { type: Number, default: 5 }, // Nouveau champ, initialisé à 5
    feedbackText: { type: String, required: false }, // Champ pour le texte du feedback
    featureSuggestions: { type: [String], required: false }, // Tableau pour les suggestions
    rating: { type: Number, required: false }, // Champ pour la note
  },

  {
    timestamps: true,
  }
);

const modelName = 'Account';

export default model<I>(modelName, instance);