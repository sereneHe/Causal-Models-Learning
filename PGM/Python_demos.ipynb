{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 59,
   "source": [
    "# from IPython.display import Image\r\n",
    "# Image('../images/2/student_full_param.png')"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "source": [
    "from pgmpy.models import BayesianModel\r\n",
    "from pgmpy.factors.discrete import TabularCPD\r\n",
    "\r\n",
    "# Defining the model structure. We can define the network by just passing a list of edges.\r\n",
    "model = BayesianModel([('G', 'H'), ('S', 'H')])\r\n",
    "\r\n",
    "# Defining individual CPDs.\r\n",
    "cpd_g = TabularCPD(variable='G', variable_card=2, values=[[0.5], [0.5]], state_names={'G': ['male', 'female']})\r\n",
    "cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.9], [0.1]], state_names={'S': ['MFF', 'no']})\r\n",
    "\r\n",
    "cpd_h = TabularCPD(variable='H', variable_card=3, \r\n",
    "                   values=[[0.3, 0.05, 0.9,  0.5],\r\n",
    "                           [0.4, 0.25, 0.08, 0.3],\r\n",
    "                           [0.3, 0.7,  0.02, 0.2]],\r\n",
    "                  evidence=['G', 'S'],\r\n",
    "                  evidence_card=[2, 2],\r\n",
    "                state_names={'G': ['male', 'female'],'S': ['MFF', 'no'],\r\n",
    "                             'H': ['long', 'short', 'none']})\r\n",
    "\r\n",
    "# Associating the CPDs with the network\r\n",
    "model.add_cpds(cpd_g, cpd_s, cpd_h)\r\n",
    "\r\n",
    "# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly \r\n",
    "# defined and sum to 1.\r\n",
    "model.check_model()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "metadata": {},
     "execution_count": 60
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "DiscreteFactor\r\n",
    "Joined Probability Distribution"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "source": [
    "import numpy as np\r\n",
    "from pgmpy.factors.discrete import JointProbabilityDistribution\r\n",
    "from pgmpy.factors.discrete import DiscreteFactor\r\n",
    "prob = DiscreteFactor(['G', 'H'], [2, 3], values=[[0.1, 0.3, 0.1],\r\n",
    "                           [0.01, 0.29, 0.2]],\r\n",
    "                           state_names={'G': ['male', 'female'],\r\n",
    "                             'H': ['none', 'short', 'long']})\r\n",
    "#prob.conditional_distribution([('G', 1)])\r\n",
    "print(prob)\r\n"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+-----------+----------+------------+\n",
      "| G         | H        |   phi(G,H) |\n",
      "+===========+==========+============+\n",
      "| G(male)   | H(none)  |     0.1000 |\n",
      "+-----------+----------+------------+\n",
      "| G(male)   | H(short) |     0.3000 |\n",
      "+-----------+----------+------------+\n",
      "| G(male)   | H(long)  |     0.1000 |\n",
      "+-----------+----------+------------+\n",
      "| G(female) | H(none)  |     0.0100 |\n",
      "+-----------+----------+------------+\n",
      "| G(female) | H(short) |     0.2900 |\n",
      "+-----------+----------+------------+\n",
      "| G(female) | H(long)  |     0.2000 |\n",
      "+-----------+----------+------------+\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "source": [
    "print(prob.marginalize(['G'],inplace=False))\r\n"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+----------+----------+\n",
      "| H        |   phi(H) |\n",
      "+==========+==========+\n",
      "| H(none)  |   0.1100 |\n",
      "+----------+----------+\n",
      "| H(short) |   0.5900 |\n",
      "+----------+----------+\n",
      "| H(long)  |   0.3000 |\n",
      "+----------+----------+\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "source": [
    "marg=prob.marginalize(['G'],inplace=False)\r\n",
    "print(prob.divide(marg,inplace=False))\r\n"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+-----------+----------+------------+\n",
      "| G         | H        |   phi(G,H) |\n",
      "+===========+==========+============+\n",
      "| G(male)   | H(none)  |     0.9091 |\n",
      "+-----------+----------+------------+\n",
      "| G(male)   | H(short) |     0.5085 |\n",
      "+-----------+----------+------------+\n",
      "| G(male)   | H(long)  |     0.3333 |\n",
      "+-----------+----------+------------+\n",
      "| G(female) | H(none)  |     0.0909 |\n",
      "+-----------+----------+------------+\n",
      "| G(female) | H(short) |     0.4915 |\n",
      "+-----------+----------+------------+\n",
      "| G(female) | H(long)  |     0.6667 |\n",
      "+-----------+----------+------------+\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "source": [
    "\r\n",
    "print(prob.reduce([('G','female')],inplace=False))"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+----------+----------+\n",
      "| H        |   phi(H) |\n",
      "+==========+==========+\n",
      "| H(none)  |   0.0100 |\n",
      "+----------+----------+\n",
      "| H(short) |   0.2900 |\n",
      "+----------+----------+\n",
      "| H(long)  |   0.2000 |\n",
      "+----------+----------+\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "source": [
    "p_HGS = DiscreteFactor(['S', 'G','H'], [ 2,2,3], \r\n",
    "                        values=[0.1, 0.3, 0.1,0.01, 0.19, 0.2,0.01, 0.06, 0.02,0.001, 0.005, 0.004\r\n",
    "                           ],\r\n",
    "                           state_names={'G': ['male', 'female'],\r\n",
    "                             'H': ['none', 'short', 'long'],\r\n",
    "                             'S': ['no', 'yes']})\r\n",
    "\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "source": [
    "\r\n",
    "print(p_HGS.reduce([('S','yes')],inplace=False))\r\n"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+-----------+----------+------------+\n",
      "| G         | H        |   phi(G,H) |\n",
      "+===========+==========+============+\n",
      "| G(male)   | H(none)  |     0.0100 |\n",
      "+-----------+----------+------------+\n",
      "| G(male)   | H(short) |     0.0600 |\n",
      "+-----------+----------+------------+\n",
      "| G(male)   | H(long)  |     0.0200 |\n",
      "+-----------+----------+------------+\n",
      "| G(female) | H(none)  |     0.0010 |\n",
      "+-----------+----------+------------+\n",
      "| G(female) | H(short) |     0.0050 |\n",
      "+-----------+----------+------------+\n",
      "| G(female) | H(long)  |     0.0040 |\n",
      "+-----------+----------+------------+\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "source": [
    "print(p_HGS.reduce([('H','long')],inplace=False))\r\n"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+--------+-----------+------------+\n",
      "| S      | G         |   phi(S,G) |\n",
      "+========+===========+============+\n",
      "| S(no)  | G(male)   |     0.1000 |\n",
      "+--------+-----------+------------+\n",
      "| S(no)  | G(female) |     0.2000 |\n",
      "+--------+-----------+------------+\n",
      "| S(yes) | G(male)   |     0.0200 |\n",
      "+--------+-----------+------------+\n",
      "| S(yes) | G(female) |     0.0040 |\n",
      "+--------+-----------+------------+\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "source": [
    "\r\n",
    "print(p_HGS.reduce([('H','long')],inplace=False).normalize(inplace=False))"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+--------+-----------+------------+\n",
      "| S      | G         |   phi(S,G) |\n",
      "+========+===========+============+\n",
      "| S(no)  | G(male)   |     0.3086 |\n",
      "+--------+-----------+------------+\n",
      "| S(no)  | G(female) |     0.6173 |\n",
      "+--------+-----------+------------+\n",
      "| S(yes) | G(male)   |     0.0617 |\n",
      "+--------+-----------+------------+\n",
      "| S(yes) | G(female) |     0.0123 |\n",
      "+--------+-----------+------------+\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "source": [
    "print(p_HGS.reduce([('H','long'),('S','yes')],inplace=False))\r\n",
    "print(p_HGS.reduce([('H','long'),('S','yes')],inplace=False).normalize(inplace=False))"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+-----------+----------+\n",
      "| G         |   phi(G) |\n",
      "+===========+==========+\n",
      "| G(male)   |   0.0200 |\n",
      "+-----------+----------+\n",
      "| G(female) |   0.0040 |\n",
      "+-----------+----------+\n",
      "+-----------+----------+\n",
      "| G         |   phi(G) |\n",
      "+===========+==========+\n",
      "| G(male)   |   0.8333 |\n",
      "+-----------+----------+\n",
      "| G(female) |   0.1667 |\n",
      "+-----------+----------+\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "source": [
    "#cpd.reorder_parents(['', 'diff'])\r\n",
    "from pgmpy.factors.discrete import DiscreteFactor\r\n",
    "\r\n",
    "cpd_gh = TabularCPD(variable='G', variable_card=2, \r\n",
    "                   values=[[0.1, 0.3, 0.1],\r\n",
    "                           [0.01, 0.29, 0.2]],\r\n",
    "                  evidence=['H'],\r\n",
    "                  evidence_card=[3],\r\n",
    "                state_names={'G': ['male', 'female'],\r\n",
    "                             'H': ['long', 'short', 'none']})\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "source": [
    "print(f\"Variables: {cpd_gh.variables}\")\r\n",
    "print(f\"Variable: {cpd_gh.variable}\")\r\n",
    "print(f\"Variable cardinality: {cpd_gh.variable_card}\")\r\n"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Variables: ['G', 'H']\n",
      "Variable: G\n",
      "Variable cardinality: 2\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "source": [
    "# We can now call some methods on the BayesianModel object.\r\n",
    "model.get_cpds()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "[<TabularCPD representing P(G:2) at 0x220f58ae8e0>,\n",
       " <TabularCPD representing P(S:2) at 0x220c86bc700>,\n",
       " <TabularCPD representing P(H:3 | G:2, S:2) at 0x220c86bc6a0>]"
      ]
     },
     "metadata": {},
     "execution_count": 72
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "source": [
    "print(model.get_cpds('H'))"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+----------+---------+---------+-----------+-----------+\n",
      "| G        | G(male) | G(male) | G(female) | G(female) |\n",
      "+----------+---------+---------+-----------+-----------+\n",
      "| S        | S(MFF)  | S(no)   | S(MFF)    | S(no)     |\n",
      "+----------+---------+---------+-----------+-----------+\n",
      "| H(long)  | 0.3     | 0.05    | 0.9       | 0.5       |\n",
      "+----------+---------+---------+-----------+-----------+\n",
      "| H(short) | 0.4     | 0.25    | 0.08      | 0.3       |\n",
      "+----------+---------+---------+-----------+-----------+\n",
      "| H(none)  | 0.3     | 0.7     | 0.02      | 0.2       |\n",
      "+----------+---------+---------+-----------+-----------+\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "source": [
    "model.get_cardinality('G')"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "metadata": {},
     "execution_count": 74
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "source": [
    "# Getting the local independencies of a variable.\r\n",
    "model.local_independencies('G')"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "(G ⟂ S)"
      ]
     },
     "metadata": {},
     "execution_count": 75
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "source": [
    "# Getting all the local independencies in the network.\r\n",
    "model.local_independencies(['G', 'S', 'H'])"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "(G ⟂ S)\n",
       "(S ⟂ G)"
      ]
     },
     "metadata": {},
     "execution_count": 76
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "source": [
    "model.active_trail_nodes('G')"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "{'G': {'G', 'H'}}"
      ]
     },
     "metadata": {},
     "execution_count": 77
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "source": [
    "model.active_trail_nodes('G', observed='H')"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "{'G': {'G', 'S'}}"
      ]
     },
     "metadata": {},
     "execution_count": 78
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "source": [
    "from pgmpy.inference import VariableElimination\r\n",
    "infer = VariableElimination(model)\r\n",
    "g_dist = infer.query(['H'])\r\n",
    "print(g_dist)"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stderr",
     "text": [
      "Eliminating: S: 100%|██████████| 2/2 [00:00<00:00, 90.97it/s]"
     ]
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+----------+----------+\n",
      "| H        |   phi(H) |\n",
      "+==========+==========+\n",
      "| H(long)  |   0.5675 |\n",
      "+----------+----------+\n",
      "| H(short) |   0.2435 |\n",
      "+----------+----------+\n",
      "| H(none)  |   0.1890 |\n",
      "+----------+----------+\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stderr",
     "text": [
      "\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "source": [
    "print(infer.query(['G'], evidence={'S': 'MFF', 'H': 'long'}))"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stderr",
     "text": [
      "\n",
      "0it [00:00, ?it/s]"
     ]
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "+-----------+----------+\n",
      "| G         |   phi(G) |\n",
      "+===========+==========+\n",
      "| G(male)   |   0.2500 |\n",
      "+-----------+----------+\n",
      "| G(female) |   0.7500 |\n",
      "+-----------+----------+\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stderr",
     "text": [
      "\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "source": [
    "infer.map_query(['G'], evidence={'S': 'MFF', 'H': 'long'})"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stderr",
     "text": [
      "Finding Elimination Order: : 100%|██████████| 2/2 [00:00<00:00, 11.77it/s]\n",
      "Finding Elimination Order: : : 0it [00:00, ?it/s]\n",
      "0it [00:00, ?it/s]\n"
     ]
    },
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "{'G': 'female'}"
      ]
     },
     "metadata": {},
     "execution_count": 81
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "Načtení souboru"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "source": [
    "from pgmpy.readwrite.XMLBeliefNetwork import XBNReader\r\n",
    "reader = XBNReader('./files/xmlbelief.xml')\r\n",
    "model = reader.get_model()\r\n",
    "print(model.nodes())"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stderr",
     "text": [
      "Finding Elimination Order: : : 0it [00:00, ?it/s]\n"
     ]
    },
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "['a', 'b', 'c', 'd', 'e']\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "source": [
    "\r\n",
    "import numpy as np\r\n",
    "\r\n",
    "edges_list = [('VisitToAsia', 'Tuberculosis'),\r\n",
    "              ('LungCancer', 'TuberculosisOrCancer'),\r\n",
    "              ('Smoker', 'LungCancer'),\r\n",
    "              ('Smoker', 'Bronchitis'),\r\n",
    "              ('Tuberculosis', 'TuberculosisOrCancer'),\r\n",
    "              ('Bronchitis', 'Dyspnea'),\r\n",
    "              ('TuberculosisOrCancer', 'Dyspnea'),\r\n",
    "              ('TuberculosisOrCancer', 'X-ray')]\r\n",
    "nodes = {'Smoker': {'States': {'no': {}, 'yes': {}},\r\n",
    "                    'role': 'chance',\r\n",
    "                    'type': 'finiteStates',\r\n",
    "                    'Coordinates': {'y': '52', 'x': '568'},\r\n",
    "                    'AdditionalProperties': {'Title': 'S', 'Relevance': '7.0'}},\r\n",
    "         'Bronchitis': {'States': {'no': {}, 'yes': {}},\r\n",
    "                        'role': 'chance',\r\n",
    "                        'type': 'finiteStates',\r\n",
    "                        'Coordinates': {'y': '181', 'x': '698'},\r\n",
    "                        'AdditionalProperties': {'Title': 'B', 'Relevance': '7.0'}},\r\n",
    "         'VisitToAsia': {'States': {'no': {}, 'yes': {}},\r\n",
    "                         'role': 'chance',\r\n",
    "                         'type': 'finiteStates',\r\n",
    "                         'Coordinates': {'y': '58', 'x': '290'},\r\n",
    "                         'AdditionalProperties': {'Title': 'A', 'Relevance': '7.0'}},\r\n",
    "         'Tuberculosis': {'States': {'no': {}, 'yes': {}},\r\n",
    "                          'role': 'chance',\r\n",
    "                          'type': 'finiteStates',\r\n",
    "                          'Coordinates': {'y': '150', 'x': '201'},\r\n",
    "                          'AdditionalProperties': {'Title': 'T', 'Relevance': '7.0'}},\r\n",
    "         'X-ray': {'States': {'no': {}, 'yes': {}},\r\n",
    "                   'role': 'chance',\r\n",
    "                   'AdditionalProperties': {'Title': 'X', 'Relevance': '7.0'},\r\n",
    "                   'Coordinates': {'y': '322', 'x': '252'},\r\n",
    "                   'Comment': 'Indica si el test de rayos X ha sido positivo',\r\n",
    "                   'type': 'finiteStates'},\r\n",
    "         'Dyspnea': {'States': {'no': {}, 'yes': {}},\r\n",
    "                     'role': 'chance',\r\n",
    "                     'type': 'finiteStates',\r\n",
    "                     'Coordinates': {'y': '321', 'x': '533'},\r\n",
    "                     'AdditionalProperties': {'Title': 'D', 'Relevance': '7.0'}},\r\n",
    "         'TuberculosisOrCancer': {'States': {'no': {}, 'yes': {}},\r\n",
    "                                  'role': 'chance',\r\n",
    "                                  'type': 'finiteStates',\r\n",
    "                                  'Coordinates': {'y': '238', 'x': '336'},\r\n",
    "                                  'AdditionalProperties': {'Title': 'E', 'Relevance': '7.0'}},\r\n",
    "         'LungCancer': {'States': {'no': {}, 'yes': {}},\r\n",
    "                        'role': 'chance',\r\n",
    "                        'type': 'finiteStates',\r\n",
    "                        'Coordinates': {'y': '152', 'x': '421'},\r\n",
    "                        'AdditionalProperties': {'Title': 'L', 'Relevance': '7.0'}}}\r\n",
    "edges = {'LungCancer': {'TuberculosisOrCancer': {'directed': 'true'}},\r\n",
    "         'Smoker': {'LungCancer': {'directed': 'true'},\r\n",
    "                    'Bronchitis': {'directed': 'true'}},\r\n",
    "         'Dyspnea': {},\r\n",
    "         'X-ray': {},\r\n",
    "         'VisitToAsia': {'Tuberculosis': {'directed': 'true'}},\r\n",
    "         'TuberculosisOrCancer': {'X-ray': {'directed': 'true'},\r\n",
    "                                  'Dyspnea': {'directed': 'true'}},\r\n",
    "         'Bronchitis': {'Dyspnea': {'directed': 'true'}},\r\n",
    "         'Tuberculosis': {'TuberculosisOrCancer': {'directed': 'true'}}}\r\n",
    "\r\n",
    "cpds = [{'Values': np.array([[0.95, 0.05], [0.02, 0.98]]),\r\n",
    "         'Variables': {'X-ray': ['TuberculosisOrCancer']}},\r\n",
    "        {'Values': np.array([[0.7, 0.3], [0.4,  0.6]]),\r\n",
    "         'Variables': {'Bronchitis': ['Smoker']}},\r\n",
    "        {'Values':  np.array([[0.9, 0.1,  0.3,  0.7], [0.2,  0.8,  0.1,  0.9]]),\r\n",
    "         'Variables': {'Dyspnea': ['TuberculosisOrCancer', 'Bronchitis']}},\r\n",
    "        {'Values': np.array([[0.99], [0.01]]),\r\n",
    "         'Variables': {'VisitToAsia': []}},\r\n",
    "        {'Values': np.array([[0.5], [0.5]]),\r\n",
    "         'Variables': {'Smoker': []}},\r\n",
    "        {'Values': np.array([[0.99, 0.01], [0.9, 0.1]]),\r\n",
    "         'Variables': {'LungCancer': ['Smoker']}},\r\n",
    "        {'Values': np.array([[0.99, 0.01], [0.95, 0.05]]),\r\n",
    "         'Variables': {'Tuberculosis': ['VisitToAsia']}},\r\n",
    "        {'Values': np.array([[1, 0, 0, 1], [0, 1, 0, 1]]),\r\n",
    "         'Variables': {'TuberculosisOrCancer': ['LungCancer', 'Tuberculosis']}}]"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "source": [
    "from pgmpy.models import BayesianModel\r\n",
    "from pgmpy.factors.discrete import TabularCPD\r\n",
    "\r\n",
    "model = BayesianModel(edges_list)\r\n",
    "\r\n",
    "for node in nodes:\r\n",
    "    for key, value in nodes[node].items():\r\n",
    "        model.nodes[node][key] = value\r\n",
    "\r\n",
    "for u in edges.keys():\r\n",
    "    for v in edges[u].keys():\r\n",
    "        #import pdb; pdb.set_trace()\r\n",
    "        for key, value in edges[u][v].items():\r\n",
    "            model.edges[(u, v)][key] = value\r\n",
    "\r\n",
    "tabular_cpds = []\r\n",
    "for cpd in cpds:\r\n",
    "    var = list(cpd['Variables'].keys())[0]\r\n",
    "    evidence = cpd['Variables'][var]\r\n",
    "    values = cpd['Values']\r\n",
    "    states = len(nodes[var]['States'])\r\n",
    "    evidence_card = [len(nodes[evidence_var]['States'])\r\n",
    "                     for evidence_var in evidence]\r\n",
    "    tabular_cpds.append(\r\n",
    "        TabularCPD(var, states, values, evidence, evidence_card))\r\n",
    "\r\n",
    "model.add_cpds(*tabular_cpds)"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "source": [
    "model.nodes()"
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "NodeView(('VisitToAsia', 'Tuberculosis', 'LungCancer', 'TuberculosisOrCancer', 'Smoker', 'Bronchitis', 'Dyspnea', 'X-ray'))"
      ]
     },
     "metadata": {},
     "execution_count": 85
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "source": [
    "from pgmpy.utils import get_example_model\r\n",
    "from pgmpy.readwrite import BIFReader, BIFWriter\r\n",
    "asia = get_example_model('alarm')\r\n",
    "writer = BIFWriter(asia)\r\n",
    "writer.write_bif(filename='alarm.bif')"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "Graphviz"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "source": [
    "import graphviz\r\n",
    "dot = graphviz.Digraph(comment='The Round Table')\r\n",
    "dot.node('A', 'King Arthur')\r\n",
    "dot.node('B', 'Sir Bedevere the Wise')\r\n",
    "dot.node('L', 'Sir Lancelot the Brave')\r\n",
    "\r\n",
    "dot.edges(['AB', 'AL'])\r\n",
    "dot.edge('B', 'L', constraint='false')\r\n",
    "print(dot.source)  "
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "// The Round Table\n",
      "digraph {\n",
      "\tA [label=\"King Arthur\"]\n",
      "\tB [label=\"Sir Bedevere the Wise\"]\n",
      "\tL [label=\"Sir Lancelot the Brave\"]\n",
      "\tA -> B\n",
      "\tA -> L\n",
      "\tB -> L [constraint=false]\n",
      "}\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "source": [
    "dot.render('test-output/round-table.gv', view=True)  "
   ],
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "'test-output\\\\round-table.gv.pdf'"
      ]
     },
     "metadata": {},
     "execution_count": 88
    }
   ],
   "metadata": {}
  }
 ],
 "metadata": {
  "hide_input": false,
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3.8.1 64-bit"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.1"
  },
  "interpreter": {
   "hash": "5714e0d71dbe446fea01dd6deda0c94d4e39f67ef1cccf2400b1c09a7a184d64"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
