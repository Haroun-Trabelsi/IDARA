import { type RequestHandler } from 'express'
import joi from '../../utils/joi'
import jwt from '../../utils/jwt'
import crypt from '../../utils/crypt'
import Account from '../../models/Account'

const login: RequestHandler = async (req, res, next) => {
  try {
    const validationError = await joi.validate(
      {
        email: joi.instance.string().email().required(),
        password: joi.instance.string().required(),
      },
      req.body,
      { stripUnknown: true } // Ignore les champs non définis dans le schéma
    )

    if (validationError) {
      return next(validationError)
    }

    const { email, password } = req.body

    // Get account from DB, and verify existence
    const account = await Account.findOne({ email })

    if (!account) {
      return next({
        statusCode: 400,
        message: 'Mauvais identifiants',
      })
    }

    // Verify password hash
    const passOk = crypt.validate(password, account.password)

    if (!passOk) {
      return next({
        statusCode: 400,
        message: 'Mauvais identifiants',
      })
    }

    // Mise à jour de lastConnexion à la date actuelle
    account.lastConnexion = new Date();
    await account.save();
    console.log(`lastConnexion mise à jour pour l'utilisateur ${account.email}: ${account.lastConnexion}`); // Log pour débogage

    // Generate access token
    const token = jwt.signToken({ uid: account._id, role: account.role })

    // Remove password from response data
    const { password: _, ...accountData } = account.toObject()

    res.status(200).json({
      message: 'Connexion réussie',
      data: accountData,
      token,
    })
  } catch (error) {
    next(error)
  }
}

export default login