import express from 'express';
import { getFtrackSession } from '../utils/session';

const router = express.Router();
const session = getFtrackSession();

router.get('/projects', async (req, res) => {
    try {
        console.log('Connected to ftrack');
        console.log('About to query projects');
        const response = await (await session).query('select id,name from Project');
        const projects = response.data;

        console.info("Listing " + projects.length + " projects");
        console.log(projects);

        res.json(projects);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to fetch projects' });
    }
});


// Get detailed info for a single project by name
// Get detailed info for a single project by name
router.get('/projects/:projectName', async (req, res) => {
  try {
    const { projectName } = req.params;

    console.log(`Fetching details for Tasks with project id : ${projectName}`);

    const sessionInstance = await session;

    // Query for the project entity with detailed fields including parent.name
    const projectQuery = await sessionInstance.query(
      `select 
        id, 
        name, 
        description, 
        status.name, 
        priority.name, 
        type.name, 
        start_date, 
        end_date, 
        time_logged, 
        bid, 
        bid_time_logged_difference, 
        created_at, 
        created_by.username, 
        project.name, 
        parent.name
      from Task 
      where project.id is ${projectName}`
    );

    console.log('Raw query data:', projectQuery.data);

    const project = projectQuery.data[0];

    if (!project) {
      return res.status(404).json({ error: 'Tasks not found' });
    }

    // Map tasks with proper nested field access and safe checks
    const tasks = projectQuery.data.map((row, index) => ({
      id: row.id,
      number: index + 1,
      type: `Task (${row.type?.name || "Unknown"})`,
      status: row.status?.name?.toLowerCase() || "pending",
      assignee: row.created_by?.username || "Unassigned",
      description: `Shot: ${row.parent?.name || "Unknown"}`,
      dueDate: row.end_date ? new Date(row.end_date).toISOString().split("T")[0] : null,
      bidHours: row.bid ? row.bid / 3600 : 0,
      actualHours: row.time_logged ? row.time_logged / 3600 : 0,
      level: 0,
      icon: row.type?.name?.toLowerCase() === "camtrack" ? "tracking" : undefined,
    }));

    console.log('Processed tasks:', tasks);

    res.json(tasks);

  } catch (err) {
    console.error('Error fetching project details:', err);
    res.status(500).json({ error: 'Failed to fetch project details' });
  }
});



export default router;
