// controllers/auth/resetPassword.ts
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

const resetPassword: RequestHandler = async (req, res, next) => {
  try {
    const validationError = await joi.validate(
      {
        token: joi.instance.string().required(),
        newPassword: joi.instance.string().min(8).required(),
      },
      req.body,
      { stripUnknown: true }
    );

    if (validationError) {
      return next(validationError);
    }

    const { token, newPassword } = req.body;

    // Vérifier et décoder le token avec typage explicite
    const decoded = jwt.verifyToken(token) as JwtPayloadWithUid | string;
    if (typeof decoded === 'string' || !decoded.uid) {
      return next({
        statusCode: 400,
        message: 'Invalid or expired reset token',
      });
    }

    const account = await Account.findById(decoded.uid);
    if (!account) {
      return next({
        statusCode: 400,
        message: 'Account not found',
      });
    }

    // Mettre à jour le mot de passe (supposant que crypt.hash est asynchrone)
    account.password = await crypt.hash(newPassword); // Utiliser await si asynchrone
    await account.save();

    res.status(200).json({
      message: 'Password reset successfully',
    });
  } catch (error) {
    next(error);
  }
};

export default resetPassword;