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

    receiveUpdates: { type: Boolean, default: false }

  },

  {
    timestamps: true,
  }
);

const modelName = 'Account';

export default model<I>(modelName, instance);