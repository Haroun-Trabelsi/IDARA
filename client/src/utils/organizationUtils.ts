import { Account } from "../@types/index";

export interface Member {
  _id: string;
  name: string;
  surname?: string;
  email: string;
  status: "pending" | "accepted" | "expired" | "AdministratorOrganization";
  invitedDate?: string;
  canInvite?: boolean;
  invitedBy?: string;
}

export const getStatusDisplay = (member: Member, account: Partial<Account> | null): string => {
  if (account?._id && member._id === account._id && account.canInvite) return "Owner";
  if (member.invitedBy && (!account?._id || member.invitedBy !== account._id)) return "-";
  switch (member.status) {
    case "pending":
      return "Invited";
    case "accepted":
      return "Invited";
    case "expired":
      return "Expired/Admin";
    case "AdministratorOrganization":
      return "Admin";
    default:
      return member.status;
  }
};

export const getStatusColor = (member: Member, account: Partial<Account> | null): string => {
  if (account?._id && member._id === account._id && account.canInvite) return "#4299e1";
  if (member.invitedBy && (!account?._id || member.invitedBy !== account._id)) return "#6b7280";
  return {
    pending: "#f59e0b",
    accepted: "#10b981",
    expired: "#ef4444",
    AdministratorOrganization: "#9b59b6",
  }[member.status] || "#6b7280";
};

export const getStatusIcon = (member: Member, account: Partial<Account> | null): string => {
  if (account?._id && member._id === account._id && account.canInvite) return "👑";
  if (member.invitedBy && (!account?._id || member.invitedBy !== account._id)) return "-";
  return {
    pending: "⏳",
    accepted: "✓",
    expired: "⚠️",
    AdministratorOrganization: "🛡️",
  }[member.status] || "?";
};

export const getAvatarInitials = (name: string, surname?: string): string =>
  `${name.charAt(0)}${surname?.charAt(0) || ""}`.toUpperCase();