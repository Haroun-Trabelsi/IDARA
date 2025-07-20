import { type RequestHandler } from 'express';
import joi from '../../utils/joi';
import jwt from '../../utils/jwt';
import crypt from '../../utils/crypt';
import Account from '../../models/Account';

const editProfile: RequestHandler = async (req, res, next) => {
  try {
    const userId = req.auth?.uid;
    if (!userId) {
      return next({
        statusCode: 401,
        message: 'Unauthorized',
      });
    }

    const validationError = await joi.validate(joi.profileSchema, req.body);
    if (validationError) {
      return next(validationError);
    }

    const { name, surname, email } = req.body;

    const existingAccount = await Account.findOne({ email, _id: { $ne: userId } });
    if (existingAccount) {
      return next({
        statusCode: 400,
        message: 'An account already exists with this email',
      });
    }

    const account = await Account.findByIdAndUpdate(
      userId,
      { name, surname, email },
      { new: true, runValidators: true }
    );

    if (!account) {
      return next({
        statusCode: 404,
        message: 'Account not found',
      });
    }

    const { password, verificationCode, verificationCodeExpires, ...data } = account.toObject();

    res.status(200).json({
      message: 'Profile updated successfully',
      account: data,
    });
  } catch (error: any) {
    next({
      statusCode: 500,
      message: 'Server error during profile update',
      error: error.message,
    });
  }
};

const changePassword: RequestHandler = async (req, res, next) => {
  try {
    const userId = req.auth?.uid;
    if (!userId) {
      return next({
        statusCode: 401,
        message: 'Unauthorized',
      });
    }

    const validationError = await joi.validate(joi.passwordSchema, req.body);
    if (validationError) {
      return next(validationError);
    }

    const { currentPassword, newPassword } = req.body;

    const account = await Account.findById(userId);
    if (!account) {
      return next({
        statusCode: 404,
        message: 'Account not found',
      });
    }

    const isMatch = await crypt.compare(currentPassword, account.password);
    if (!isMatch) {
      return next({
        statusCode: 400,
        message: 'Current password is incorrect',
      });
    }

    const hash = await crypt.hash(newPassword);
    account.password = hash;
    await account.save();

    res.status(200).json({
      message: 'Password changed successfully',
    });
  } catch (error: any) {
    next({
      statusCode: 500,
      message: 'Server error during password change',
      error: error.message,
    });
  }
};

export { editProfile, changePassword };