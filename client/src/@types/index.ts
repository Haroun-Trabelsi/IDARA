export interface Account {
  _id: string;
  name: string;
  surname: string;
  email: string;
  password: string;
  role: 'user' | 'admin';
  isVerified: boolean;
  verificationCode?: string;
  verificationCodeExpires?: Date;
  mfaSecret?: string;
  mfaEnabled: boolean;
  organizationName: string; // Ajout du champ
  invitedBy?: string; // Référence à l'ID de l'utilisateur qui a invité, optionnel
  canInvite: boolean; // Permission d'inviter, par défaut false sauf pour l'admin initial
  status: "pending" | "accepted" | "expired" | "AdministratorOrganization";
  invitedDate?: string;
  mustCompleteProfile: boolean; // Nouveau champ
  receiveUpdates: boolean;
  feedbackText?: string;
  featureSuggestions?: string[];
  rating?: number;
  teamSize: string;
  region: string;
  ID_Organization: String;

}

export interface FormData {
  name?: string;
  surname?: string;
  organizationName: string;
  email: string;
  password: string;
  teamSize: string;
  region: string;
  //receiveUpdates: boolean;

}