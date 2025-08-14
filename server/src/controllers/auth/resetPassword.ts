import { type RequestHandler } from 'express';
import joi from '../../utils/joi';
import jwt from '../../utils/jwt';
import crypt from '../../utils/crypt';
import Account from '../../models/Account';

// Définir un type pour le payload du token
interface JwtPayloadWithUid {
  uid: string;
  iat?: number;
  exp?: number;
}

const resetPassword: RequestHandler = async (req, res): Promise<void> => {
  console.log('Requête de réinitialisation de mot de passe reçue :', req.body);

  const validationError = await joi.validate(
    {
      token: joi.instance.string().required(),
      newPassword: joi.instance.string().min(8).required(),
    },
    req.body,
    { stripUnknown: true }
  );

  if (validationError) {
    console.log('Erreur de validation :', validationError);
    res.status(400).json({
      message: validationError.message,
      error: validationError.message,
    });
    return;
  }

  const { token, newPassword } = req.body;

  // Vérifier et décoder le token avec typage explicite
  let decoded: JwtPayloadWithUid;
  try {
    decoded = jwt.verifyToken(token) as JwtPayloadWithUid;
    console.log('Debug: Verified token payload:', decoded);
  } catch (error) {
    console.log('Jeton invalide ou expiré :', token);
    res.status(400).json({
      message: 'Invalid or expired reset token. Redirecting to error page.',
      redirect: 'http://localhost:3000/error',
    });
    return;
  }

  if (!decoded.uid) {
    console.log('Jeton invalide, pas de UID :', token);
    res.status(400).json({
      message: 'Invalid reset token. Redirecting to error page.',
      redirect: 'http://localhost:3000/error',
    });
    return;
  }

  const account = await Account.findOne({
    _id: decoded.uid,
    resetPasswordToken: token,
    verificationTokenExpires: { $gt: new Date() },
  });

  if (!account) {
    console.log('Jeton déjà utilisé, expiré ou compte non trouvé pour le jeton :', token);
    console.log('Debug: Recherche dans la base - uid:', decoded.uid, 'resetPasswordToken:', token);
    res.status(400).json({
      redirect: 'http://localhost:3000/error',
    });
    return;
  }

  try {
    // Mettre à jour le mot de passe
    account.password = await crypt.hash(newPassword);
    account.resetPasswordToken = undefined; // Supprimer le jeton après utilisation
    account.verificationTokenExpires = undefined; // Supprimer l'expiration
    await account.save();
    console.log('Mot de passe réinitialisé avec succès pour le compte :', account._id);
    console.log('Debug: Après réinitialisation - resetPasswordToken:', account.resetPasswordToken);

    res.status(200).json({
      message: 'Password reset successfully. You can now log in with your new password.',
    });
  } catch (error: any) {
    console.error('Erreur lors de la réinitialisation du mot de passe :', error);
    res.status(500).json({
      message: 'Server error during password reset',
      error: error.message,
    });
  }
};

export default resetPassword;