import { type RequestHandler } from 'express';
import joi from '../../utils/joi';
import jwt from '../../utils/jwt';
import crypt from '../../utils/crypt';
import { sendVerificationEmail } from '../../utils/nodemailer';
import Account from '../../models/Account';
import crypto from 'crypto';

const register: RequestHandler = async (req, res, next) => {
  try {
    const validationError = await joi.validate(
      {
        name: joi.instance.string().required(),
        surname: joi.instance.string().required(),
        organizationName: joi.instance.string().required(),
        email: joi.instance.string().email().required(),
        password: joi.instance.string().required(),
        role: joi.instance.string().valid('user', 'admin').default('user'),
        canInvite: joi.instance.boolean().default(false),
        isVerified: joi.instance.boolean().default(false),
        mfaEnabled: joi.instance.boolean().default(false),
      },
      req.body,
      { stripUnknown: true }
    );

    if (validationError) {
      return next(validationError);
    }

    const { name, surname, organizationName, email, password, role, canInvite, isVerified, mfaEnabled } = req.body;

    const found = await Account.findOne({ email });

    if (found) {
      return next({
        statusCode: 400,
        message: 'An account already exists with this email',
      });
    }

    const hash = await crypt.hash(password);

    const verificationCode = crypto.randomInt(100000, 999999).toString();
    const verificationCodeExpires = new Date(Date.now() + 30 * 1000);
    console.log('Registering user:', { email, verificationCode, verificationCodeExpires });

    const account = new Account({
      name,
      surname,
      organizationName,
      email,
      password: hash,
      role,
      isVerified,
      verificationCode,
      verificationCodeExpires,
      mfaEnabled,
      canInvite,
    });
    await account.save();

    await sendVerificationEmail(email, name, verificationCode);

    const token = jwt.signToken({ uid: account._id, role: account.role });

    const { password: _password, verificationCode: _code, verificationCodeExpires: _expires, ...data } = account.toObject();

    // Réponse avec email pour stockage local
    res.status(201).json({
      message: 'Signup successful! Please check your email for a verification code.',
      data: { ...data, email }, // Inclure email dans la réponse
      token,
    });
  } catch (error: any) {
    if (error.code === 11000) {
      return next({
        statusCode: 400,
        message: 'An account already exists with this email',
      });
    }
    next({
      statusCode: 500,
      message: 'Server error during signup',
      error: error.message,
    });
  }
};

export default register;