import { type RequestHandler } from 'express'
import speakeasy from 'speakeasy'
import QRCode from 'qrcode'
import Account from '../../models/Account'

const setupMFA: RequestHandler = async (req, res, next) => {
  try {
    const { email } = req.body
    if (!email) {
      return next({
        statusCode: 400,
        message: 'Email is required',
      })
    }

    const account = await Account.findOne({ email })
    if (!account) {
      return next({
        statusCode: 404,
        message: 'Account not found',
      })
    }

    // Générer une clé secrète avec speakeasy
    const secret = speakeasy.generateSecret({
      name: `YourApp:${email}`, // Nom de l'application et email de l'utilisateur
      issuer: 'YourApp', // Nom de ton application
    })

    // Vérifier que otpauth_url est défini
    if (!secret.otpauth_url) {
      throw new Error('Failed to generate OTP authentication URL')
    }

    // Générer le QR code à partir de l'URL OTAuth
    const qrCodeUrl = await QRCode.toDataURL(secret.otpauth_url)

    // Sauvegarder la clé secrète dans le compte (non activée tant que non vérifiée)
    account.mfaSecret = secret.base32
    await account.save()

    res.status(200).json({
      qrCode: qrCodeUrl,
      secret: secret.base32, // Envoie la clé au frontend pour débogage (à retirer en production)
    })
  } catch (error) {
    return next({
      statusCode: 500,
      message: 'Server error during MFA setup',
      error: error instanceof Error ? error.message : 'Unknown error',
    })
  }
}

const verifyMFA: RequestHandler = async (req, res, next) => {
  try {
    const { email, code, secret } = req.body
    if (!email || !code || !secret) {
      return next({
        statusCode: 400,
        message: 'Email, code, and secret are required',
      })
    }

    const account = await Account.findOne({ email })
    if (!account || !account.mfaSecret) {
      return next({
        statusCode: 400,
        message: 'MFA not configured for this account',
      })
    }

    // Vérifier le code TOTP avec speakeasy
    const verified = speakeasy.totp.verify({
      secret: account.mfaSecret,
      encoding: 'base32',
      token: code,
      window: 1, // Tolérance de 30 secondes (1 fenêtre de chaque côté)
    })

    if (verified) {
      account.mfaEnabled = true
      await account.save()
      res.status(200).json({ message: 'MFA enabled successfully' })
    } else {
      return next({
        statusCode: 400,
        message: 'Invalid MFA code',
      })
    }
  } catch (error) {
    return next({
      statusCode: 500,
      message: 'Server error during MFA verification',
      error: error instanceof Error ? error.message : 'Unknown error',
    })
  }
}

export { setupMFA, verifyMFA }