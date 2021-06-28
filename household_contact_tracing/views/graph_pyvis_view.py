import datetime

from pyvis.network import Network as pvNetwork
import networkx as nx
import os
import webbrowser
from bs4 import BeautifulSoup as bs

from household_contact_tracing.views.branching_process_view import BranchingProcessView
from household_contact_tracing.branching_process_model import BranchingProcessModel


class GraphPyvisView(BranchingProcessView):
    """
        Graph view for visually displaying network as a Pyvis graph.
        Uses Pyvis library: https://pyvis.readthedocs.io/en/latest/tutorial.html

        Attributes
        ----------
            _model (BranchingProcessModel):
                The branching process model who's data is being displayed to the user
            _open_in_browser (bool):
                Whether to open a browser tab and display the graph
            _filename (str):
                The file spec used to save the output file.


        Methods
        -------

            set_display(self, display: bool)
                choose whether to show these 'shell' (text printouts) to the user

            set_show_increment_graphs(self, show_all):
                Sets whether to display this graph view every time the graph is incremented.

            graph_change(self, subject: BranchingProcessModel)
                Respond to changes in graph (nodes/households network)

            model_state_change(self, subject: BranchingProcessModel):
                Respond to changes in model state (e.g. running, extinct, timed-out)

            model_step_increment(self, subject: BranchingProcessModel):
                Respond to increment in simulation

            model_simulation_stopped(self, subject: BranchingProcessModel)
                Respond to end of simulation run

    """
    def __init__(self, model: BranchingProcessModel):
        """
        Constructor for GraphPyvisView

            Parameters:
                model (BranchingProcessModel): The branching process model who's data is being displayed to the user

            Returns:
                new GraphPyvisView
        """
        self._model = model

        self.filename = os.path.join(os.path.dirname(self._model.root_dir),
                                     'temp',
                                     'pyvis_graph_{}.html'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self._open_in_browser = False

        # Register as observer
        self._model.register_observer_state_change(self)
        self._model.register_observer_simulation_stopped(self)

    @property
    def open_in_browser(self):
        """ Get open_in_browser (whether to open a browser tab and display the graph) """
        return self._open_in_browser

    @open_in_browser.setter
    def open_in_browser(self, open_in_browser: bool):
        """ Set open_in_browser (whether to open a browser tab and display the graph) """
        self._open_in_browser = open_in_browser

    def set_display(self, show: bool):
        """
        Sets whether this pyvis graph view is displayed or not.

            Parameters:
                show (bool): To display this view, set to True

            Returns:
                None
        """
        if show:
            self._model.register_observer_simulation_stopped(self)
        else:
            self._model.remove_observer_simulation_stopped(self)

    def set_show_increment_graphs(self, show_all):
        """
        Sets whether to display this graph view every time the graph is incremented.

            Parameters:
                show_all (bool): To display this view, set to True

            Returns:
                None
        """
        if show_all:
            self._model.register_observer_graph_change(self)
        else:
            self._model.remove_observer_graph_change(self)

    def model_state_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in model state (e.g. running, extinct, timed-out)

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    def model_step_increment(self, subject: BranchingProcessModel):
        """
        Respond to single step increment in simulation

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    def model_simulation_stopped(self, subject: BranchingProcessModel):
        """
        Respond to end of simulation run

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        if self not in subject.observers_graph_change:
            self.draw_network(subject)

    def graph_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in graph (nodes/households network)

            Parameters:
                subject (SimulationModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    def draw_network(self, model: BranchingProcessModel):
        """
        Draws the network generated by the model using Pyvis and saves to temp directory.
        (Also displays in a new browser tab if self._open_in_browser == True)

            Parameters:
                model (SimulationModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        network = model.network

        nt = pvNetwork('1000px', '1000px')

        reduced_graph = self._adapt_nodes(network.graph)
        # populates the nodes and edges data structures
        nt.from_nx(reduced_graph)

        # The following 2 rows are potentially useful for configuring the pyvis graph: They are intended as temporary
        # for parameter setting only.
        # Warning: If you do (temporarily) uncomment out either of the following lines, the parameter control
        #   clashes with the legend (if _add_legend() is called).
        #  nt.show_buttons(filter_=['physics'])
        #  nt.show_buttons()

        nt.save_graph(self.filename)
        # Add the legend to the saved html file
        self._add_legend()

        if self.open_in_browser:
            webbrowser.open(self.filename)

    def _adapt_nodes(self, graph: nx.Graph):
        """
        Adapts the network graph (nx.graph) to correct format for pyvis display.

            Parameters:
                graph (networkx.Graph): The network graph to be copied and adjusted.

            Returns:
                graph (networkx.Graph)
        """

        result = graph.copy()
        network = self._model.network

        for node in list(result.nodes(data=True)):
            # Add the required info to the graph for rendering
            result.nodes[node[0]]['label'] = node[1]['node_obj'].household.id
            result.nodes[node[0]]['group'] = node[1]['node_obj'].node_type().value
            result.nodes[node[0]]['color'] = network.node_colours[node[1]['node_obj'].node_type()].colour
            result.nodes[node[0]]['title'] = network.node_colours[node[1]['node_obj'].node_type()].label

            # Remove the Node object from the graph node, as pyvis can't handle non JSON-serialisable objects
            node[1].pop('node_obj')

        for edge in result.edges.data():
            # For full set of edge settings, see: https://visjs.github.io/vis-network/docs/network/edges.html
            # Add the required info to the graph for rendering
            edge[2]['color'] = network.edge_colours[edge[2]['edge_type']].colour
            edge[2]['title'] = network.edge_colours[edge[2]['edge_type']].label

            # Too crowded (diagram) with labels on edges, but leaving in, in case required
            # edge[2]['label'] = edge_colours[edge[2]['edge_type']].label
            # edge[2]['font.size'] = '8'

            # Remove the Node object from the graph node, as pyvis can't handle non JSON-serialisable objects
            edge[2].pop('edge_type')

        return result

    def _add_legend(self):
        """ Add a legend to the HTML graph output """
        # load the file
        with open(self.filename) as inf:
            txt = inf.read()
            soup = bs(txt, "html.parser")

        self._create_html_table(self._model.network.edge_colours, soup, 'Edges')
        self._create_html_table(self._model.network.node_colours, soup, 'Nodes')

        # save the file again
        with open(self.filename, "w") as out_file:
            out_file.write(str(soup))

    @staticmethod
    def _create_html_table(network_colour_dict: dict, soup: bs, title: str):
        """ Add a (HTML) table to the legend for the HTML graph output """
        # create new link
        new_table = soup.new_tag("table")
        # insert it into the document
        soup.body.append(new_table)

        # Edges legend
        new_row = soup.new_tag('tr')
        new_table.append(new_row)
        new_cell = soup.new_tag('th')
        new_row.append(new_cell)
        new_cell.string = title

        for colour in network_colour_dict:
            new_row = soup.new_tag('tr')
            new_table.append(new_row)
            new_cell = soup.new_tag('td')
            new_row.append(new_cell)
            new_cell.string = network_colour_dict[colour].label
            new_cell = soup.new_tag('td')
            new_row.append(new_cell)
            new_cell['style'] = 'background-color:{}; width: 20px;'.format(network_colour_dict[colour].colour)
