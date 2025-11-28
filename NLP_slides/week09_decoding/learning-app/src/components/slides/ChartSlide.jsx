/**
 * Chart Slide - Type 3
 * Data visualization with Recharts
 */

import { Box, Typography, Paper } from '@mui/material';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';

const ChartSlide = ({ title, chartType = 'bar', data, dataKey, xKey = 'name', description }) => {
  const ChartComponent = chartType === 'bar' ? BarChart : LineChart;
  const DataComponent = chartType === 'bar' ? Bar : Line;

  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <Box>
        <Typography variant="h4" color="primary" gutterBottom fontWeight="bold" sx={{ mb: 3 }}>
          {title}
        </Typography>
        {description && (
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            {description}
          </Typography>
        )}
        <Paper elevation={2} sx={{ p: 3 }}>
          <ResponsiveContainer width="100%" height={400}>
            <ChartComponent data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey={xKey} />
              <YAxis />
              <Tooltip />
              <Legend />
              <DataComponent
                dataKey={dataKey}
                fill="#3333B2"
                stroke="#3333B2"
                strokeWidth={2}
              />
            </ChartComponent>
          </ResponsiveContainer>
        </Paper>
      </Box>
    </motion.div>
  );
};

export default ChartSlide;
