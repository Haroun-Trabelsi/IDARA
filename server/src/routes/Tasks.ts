import { Router } from 'express'
import Result from '../models/Result'

const router = Router()

router.get('/results_by_task', async (req, res, next) => {
  try {
    const project = req.query.project as string

    if (!project) {
      return res.status(400).json({ success: false, message: "Missing 'project' query parameter" })
    }

    // ✅ Query only documents for the requested project AND version = 3
    const allResults = await Result.find({ project, version: 4 })

    const structured: Record<string, Record<string, Record<string, any>>> = {}

    for (const doc of allResults) {
      const { sequence, shot, classification_result, predicted_vfx_hours } = doc

      if (!structured[sequence]) structured[sequence] = {}
      if (!structured[sequence][shot]) structured[sequence][shot] = {}

      structured[sequence][shot] = {
        predicted_class: classification_result?.predicted_class || '',
        probabilities: classification_result?.probabilities || {},
        predicted_hours: predicted_vfx_hours ?? null, // ✅ Include predicted hours
      }
    }

    res.status(200).json({ success: true, results: structured })
  } catch (error) {
    next(error)
  }
})

export default router
