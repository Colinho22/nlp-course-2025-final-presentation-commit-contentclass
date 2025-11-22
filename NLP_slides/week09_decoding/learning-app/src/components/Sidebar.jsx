/**
 * Sidebar Component - 280px fixed width
 * Shows 3 learning goals with icons, progress, and click navigation
 */

import { Drawer, Box, Typography, List, ListItemButton, ListItemIcon, ListItemText, LinearProgress, Chip } from '@mui/material';
import { CheckCircle, RadioButtonUnchecked, ArrowForward } from '@mui/icons-material';
import { learningGoals, calculateGoalProgress } from '../data/learningGoals';

const Sidebar = ({ currentSlide, onGoalClick, completedSlides }) => {
  const getCurrentGoal = () => {
    return learningGoals.findIndex(goal =>
      currentSlide >= goal.slideRange[0] - 1 && currentSlide <= goal.slideRange[1] - 1
    );
  };

  const currentGoalIndex = getCurrentGoal();

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: 280,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 280,
          boxSizing: 'border-box',
        },
      }}
    >
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" color="primary" gutterBottom fontWeight="bold">
          Learning Goals
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 2 }}>
          Week 9: Decoding Strategies
        </Typography>

        <List>
          {learningGoals.map((goal, index) => {
            const progress = calculateGoalProgress(goal, completedSlides);
            const isComplete = progress === 100;
            const isCurrent = index === currentGoalIndex;

            return (
              <ListItemButton
                key={goal.id}
                selected={isCurrent}
                onClick={() => onGoalClick(goal.slideRange[0] - 1)}
                sx={{ mb: 1, borderRadius: 1 }}
              >
                <ListItemIcon>
                  {isComplete ? (
                    <CheckCircle color="success" />
                  ) : isCurrent ? (
                    <ArrowForward color="primary" />
                  ) : (
                    <RadioButtonUnchecked />
                  )}
                </ListItemIcon>

                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <span>{goal.icon}</span>
                      <Typography variant="body2" fontWeight={isCurrent ? 'bold' : 'normal'}>
                        {goal.title}
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <Box sx={{ mt: 0.5 }}>
                      <Typography variant="caption" display="block" color="text.secondary">
                        {goal.slides} slides
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={progress}
                        sx={{ mt: 0.5, height: 6, borderRadius: 3 }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        {progress}%
                      </Typography>
                    </Box>
                  }
                />
              </ListItemButton>
            );
          })}
        </List>

        {/* Overall Progress */}
        <Box sx={{ mt: 3, p: 2, bgcolor: 'primary.light', borderRadius: 1 }}>
          <Typography variant="caption" fontWeight="bold" color="primary.dark">
            OVERALL PROGRESS
          </Typography>
          <LinearProgress
            variant="determinate"
            value={Math.round((Object.keys(completedSlides).length / 62) * 100)}
            sx={{ my: 1 }}
          />
          <Typography variant="caption" color="primary.dark">
            {Object.keys(completedSlides).length} / 62 slides
          </Typography>
        </Box>
      </Box>
    </Drawer>
  );
};

export default Sidebar;
