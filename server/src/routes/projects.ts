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
    console.log(`Fetching tasks for project ID: ${projectName}`);

    const sessionInstance = await session;

    // 1. Get task data with assignments (only user IDs here)
    const taskQuery = await sessionInstance.query(`
      select 
        id,
        name,
        type.name,
        status.name,
        parent.name,
        parent.parent.name,
        project.name,
        created_at,
        start_date,
        end_date,
        time_logged,
        bid,
        assignments.resource.id
      from Task
      where project.id is ${projectName}
    `);

    const tasksRaw = taskQuery.data;

    // 2. Collect unique user IDs from assignments
    const userIdSet = new Set<string>();
    tasksRaw.forEach(task => {
      if (Array.isArray(task.assignments)) {
        task.assignments.forEach(assign => {
          if (assign.resource?.id) userIdSet.add(assign.resource.id);
        });
      }
    });

    const userIds = Array.from(userIdSet);

    // 3. Query user details
    let usersById: Record<string, string> = {};
    if (userIds.length > 0) {
      const userQuery = await sessionInstance.query(`
        select id, first_name, last_name, username from User where id in (${userIds.join(',')})
      `);
      usersById = Object.fromEntries(
        userQuery.data.map((u: any) => [
          u.id,
          u.first_name || u.last_name
            ? `${u.first_name || ''} ${u.last_name || ''}`.trim()
            : u.username,
        ])
      );
    }

    // 4. Format tasks with mapped assignees
    const tasks = tasksRaw.map((row, index) => {
      const assignees =
        Array.isArray(row.assignments) && row.assignments.length > 0
          ? row.assignments
              .map(a => usersById[a.resource?.id] || null)
              .filter(Boolean)
              .join(', ')
          : 'Unassigned';

      return {
        id: row.id,
        number: index + 1,
        type: `Task (${row.type?.name || 'Unknown'})`,
        status: row.status?.name?.toLowerCase() || 'pending',
        assignee: assignees,
        sequence: row.parent?.parent?.name || 'Unassigned',
        description: row.parent?.name || 'Unknown',
        dueDate: row.end_date
          ? new Date(row.end_date).toISOString().split('T')[0]
          : null,
        bidHours: row.bid ? row.bid / 3600 : 0,
        actualHours: row.time_logged ? row.time_logged / 3600 : 0,
        level: 0,
        icon: row.type?.name?.toLowerCase() === 'camtrack' ? 'tracking' : undefined,
      };
    });

    console.log('Final tasks with assignees:', tasks);
    res.json(tasks);
  } catch (err) {
    console.error('Error fetching project details:', err);
    res.status(500).json({ error: 'Failed to fetch project details' });
  }
});





export default router;
