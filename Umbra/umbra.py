# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Umbra
                                 A QGIS plugin
 Unstructured mesh builder
                              -------------------
        begin                : 2015-11-10
        git sha              : $Format:%H$
        copyright            : (C) 2015 by Rusty Holleman
        email                : rustychris@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from __future__ import print_function
from __future__ import absolute_import
import os
import six

import logging
log=logging.getLogger('umbra') # setup in __init__.py

from qgis.PyQt.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, Qt
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
# Initialize Qt resources from file resources.py
from . import resources

from qgis.core import QgsProject, Qgis

from stompy.grid import unstructured_grid
six.moves.reload_module(unstructured_grid)

# Import the code for the DockWidget
from .umbra_dockwidget import UmbraDockWidget
from .grid_info import GridInfo

import os.path

# Import the dialogs:
from . import (umbra_openlayer, umbra_savelayer, umbra_newlayer, combine_grids,
               quad_generator,umbra_grid_properties)

from . import umbra_layer
from . import umbra_editor_tool

scope="Umbra" # for reading/writing project files

class Boiler(object):
    """QGIS Plugin Implementation."""
    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        log.info("** Top of Boiler.__init__")
        # Save reference to the QGIS interface
        self.iface = iface

        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)

        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'Umbra_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Umbra')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'Umbra')
        self.toolbar.setObjectName(u'Umbra')

        log.info("Boiler: ** INITIALIZING Umbra")

        self.pluginIsActive = False 
        self.dockwidget = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('Umbra', message)


    def add_action(self,
                   icon_path,
                   text,
                   callback,
                   enabled_flag=True,
                   add_to_menu=True,
                   add_to_toolbar=True,
                   status_tip=None,
                   whats_this=None,
                   parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action


the_umbra=None

class Umbra(Boiler):
    """
    Core more specific to the Umbra plugin
    """
    def __init__(self, iface):
        global the_umbra
        the_umbra=self

        self.log=log

        self.log.info('Firing up Umbra')
        super(Umbra,self).__init__(iface)
        self.log.info('just called Boiler __init__')

        self.canvas=self.iface.mapCanvas()
        self.gridlayers=[]

        self.openlayer_state={}

        QgsProject.instance().writeProject.connect(self.on_writeProject)
        iface.projectRead.connect(self.on_readProject)

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        print("** initGui")

        icon_path = ':/plugins/Umbra/icon.png'
        # self.add_action(
        #     icon_path,
        #     text=self.tr(u'Umbra'),
        #     callback=self.run,
        #     parent=self.iface.mainWindow())

        self.editor_tool = umbra_editor_tool.UmbraEditorTool(self.iface,umbra=self)

        self.add_action(icon_path,text='Open Umbra layer',
                        callback=self.open_layer,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)

        self.add_action(icon_path,text='New Umbra layer',
                        callback=self.new_layer,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)

        self.add_action(icon_path,text='Save Umbra layer',
                        callback=self.save_layer,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)
        self.add_action(icon_path,text='Renumber nodes/edges/cells',
                        callback=self.renumber_layer,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)

        self.add_action(icon_path,text='Edit grid properties',
                        callback=self.edit_grid_properties,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)
        
        self.add_action(icon_path,text="Combine grids",
                        callback=self.show_combine_grids,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)
        
        self.add_action(icon_path,text='Show cell centers',
                        callback=self.show_cell_centers,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)

        self.add_action(icon_path,text='Show Umbra panel',
                        callback=self.dockwidget_show,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)

        self.add_action(icon_path,text='Grid information',
                        callback=self.show_grid_info,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)
        
        self.add_action(icon_path,text='Open quad generation',
                        callback=self.show_quad_generator,
                        parent=self.iface.mainWindow(),
                        add_to_menu=True,
                        add_to_toolbar=False)
        
        self.add_action(icon_path,text='Mesh Edit',
                        callback=self.enable_tool,
                        parent=self.iface.mainWindow(),
                        add_to_menu=False,
                        add_to_toolbar=True)

    #--------------------------------------------------------------------------

    # def on_close_dockwidget(self):
    #     """
    #     record that the dockwidget was closed
    #     """
    # 
    #     log.info("** cleaning up dockwidget")
    # 
    #     # disconnects
    #     self.dockwidget.closingDockWidget.disconnect(self.on_close_dockwidget)
    #     self.dockwidget=None
    #     log.info("disconnected")

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI.
        should remove any plugin state, layers, etc., 
        should end with reversing the steps of initGui
        """

        self.log.info("** UNLOAD Umbra")

        # Try to disconnect ...
        QgsProject.instance().writeProject.disconnect(self.on_writeProject)
        self.iface.projectRead.disconnect(self.on_readProject)

        self.log.info("** disconnected project signals")

        # remove any umbra layers - doesn't seem to be working.
        for gridlayer in self.gridlayers:
            gridlayer.remove_all_qlayers()

        self.log.info("** attempted to remove qlayers")

        try:
            for action in self.actions:
                self.iface.removePluginMenu(
                    self.tr(u'&Umbra'),
                    action)
                self.iface.removeToolBarIcon(action)
        except Exception as exc:
            self.log.error("While removing toolbaricon")
            self.log.error(str(exc))

        # remove the toolbar(?)
        self.toolbar=None

        # decommission the editor
        self.editor_tool.unload()
        self.editor_tool=None

        self.dockwidget_hide() # ideally really remove it, but maybe good enough to just hide.

        self.log.info("** done with UNLOAD Umbra")

    #--------------------------------------------------------------------------

    def activate(self):
        if not self.pluginIsActive:
            self.pluginIsActive = True

            log.info("** STARTING Umbra")
            self.dockwidget_show()

            # if this is omitted, be sure to also skip over
            # the disconnects.
            # this doesn't seem to really work, and leads to a bunch
            # of callbacks lying around...
            #li=self.iface.legendInterface()
            #li.currentLayerChanged.connect(self.on_layer_changed)

    def dockwidget_show(self):
            # dockwidget may not exist if:
            #    first run of plugin
            #    removed on close (see self.onClosePlugin method)
            if self.dockwidget == None:
                # Create the dockwidget (after translation) and keep reference
                self.dockwidget = UmbraDockWidget(umbra=self)

                # connect to provide cleanup on closing of dockwidget
                # self.dockwidget.closingDockWidget.connect(self.on_close_dockwidget)

                # show the dockwidget
                self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dockwidget)
            self.dockwidget.show()
            print("Done starting")

    def dockwidget_hide(self):
        if self.dockwidget is not None:
            self.dockwidget.hide()

    def current_layer_is_umbra(self,clayer=None):
        return self.active_gridlayer(clayer=clayer) is not None

    def current_grid(self):
        gridlayer=self.active_gridlayer()
        if gridlayer is not None:
            return gridlayer.grid
        else:
            log.warning("request for current grid, but umbra didn't find one")
            return None

    def open_layer(self):
        self.activate()
        dialog=umbra_openlayer.UmbraOpenLayer(parent=self.iface.mainWindow(),
                                              iface=self.iface,
                                              umbra=self)
        dialog.exec_()

    def new_layer(self):
        self.activate()
        dialog=umbra_newlayer.UmbraNewLayer(parent=self.iface.mainWindow(),
                                            iface=self.iface,
                                            umbra=self)
        dialog.exec_()

    def active_gridlayer(self,clayer=None,multiple=False):
        """
        Check either the currently selected layer, or a given clayer(QLayer)
        to find the corresponding UmbraLayer instance.  Return None if no
        match is possible.
        multiple: if True, return a list of all matches
        clayer can be a QLayer or list thereof
        """
        if not self.pluginIsActive:
            log.info("active_gridlayer: plugin not active")
            return None

        if clayer is None:
            clayer=self.iface.layerTreeView().selectedLayers()
            if len(clayer)==0:
                # NB: possible that a group is selected here, but we
                # have no way of checking for that.
                if multiple:
                    return []
                else:
                    return None
        elif not isinstance(clayer,list):
            clayer=[clayer]

        log.info("active_gridlayer: Searching for layer " + str(clayer))

        hits=[]
        # This could be sped up with a hash table, though the track record
        # for keeping state in sync has not been good, so stick with this slower
        # approach for now.
        for gridlayer in self.gridlayers:
            for one_layer in clayer:
                if gridlayer.match_to_qlayer(one_layer):
                    self.log.info('  yep - matched %s'%gridlayer)
                    if multiple:
                        hits.append(gridlayer)
                    else:
                        return gridlayer

        if multiple:
            self.log.info('  %d hits'%len(hits))
            return hits
        else:
            self.log.info('  nope, no match')
            return None
    
    def show_combine_grids(self):
        dialog=combine_grids.CombineGrids(parent=self.iface.mainWindow(),
                                          iface=self.iface,
                                          umbra=self)
        dialog.exec_()

    def save_layer(self):
        glayer = self.active_gridlayer()
        if glayer is None:
            self.iface.messageBar().pushMessage("Select layer!",
                                                "No umbra layer selected",
                                                level=Qgis.Info, duration=3)
            return

        # TODO: probably this gets routed through UmbraLayer?
        dialog=umbra_savelayer.UmbraSaveLayer(parent=self.iface.mainWindow(),
                                              iface=self.iface,
                                              umbra=self)
        dialog.exec_()

    def edit_grid_properties(self):
        glayer = self.active_gridlayer()
        if glayer is None:
            self.iface.messageBar().pushMessage("Select layer!",
                                                "No umbra layer selected",
                                                level=Qgis.Info, duration=3)
            return

        dialog=umbra_grid_properties.UmbraGridProperties(parent=self.iface.mainWindow(),
                                                         iface=self.iface,
                                                         layer=glayer,
                                                         umbra=self)
        dialog.exec_()

    def renumber_layer(self):
        glayer = self.active_gridlayer()
        if glayer is None:
            self.iface.messageBar().pushMessage("Select layer!",
                                                "No umbra layer selected",
                                                level=Qgis.Info, duration=3)
            return
        glayer.renumber()
        # not strictly renumbering, but generally you'd want to drop
        # unneeded nodes at the same time.
        glayer.grid.delete_orphan_nodes()

        self.iface.messageBar().pushMessage("Done", "Renumbering is complete", level=Qgis.Success, duration=3)
    def show_grid_info(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            dialog=GridInfo(self,glayer)
            dialog.exec_()

    quad_generator=None
    def show_quad_generator(self):
        if self.quad_generator is None:
            self.quad_generator=quad_generator.QuadLaplacian(self)
            # Can I do this without adding it to the iface?
        self.quad_generator.show()

    def show_cell_centers(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            glayer.add_centers_layer()

    def set_cell_quality_style(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            self.log.info("set_cell_quality_style: found layer, passing on to it")
            glayer.set_cell_quality_style()

    def set_edge_quality_style(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            glayer.set_edge_quality_style()

    def add_node_layer(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            glayer.add_layer_by_tag(tag='nodes')
    def add_edge_layer(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            glayer.add_layer_by_tag(tag='edges')
    def add_cell_layer(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            glayer.add_layer_by_tag(tag='cells')
    def add_centers_layer(self):
        glayer = self.active_gridlayer()
        if glayer is not None:
            glayer.add_layer_by_tag(tag='centers')

    def merge_grids(self):
        selected=self.active_gridlayer(multiple=True)
        if len(selected)<2:
            self.log.info("Need at least two selected grid layers to merge")
            return

        gA=selected[0].grid.copy()
        for l in selected[1:]:
            gB=l.grid
            if gB.max_sides>gA.max_sides:
                gA,gB=gB.copy(),gA
            gA.add_grid(gB,merge_nodes='auto')

        # Add this grid
        my_layer = umbra_layer.UmbraLayer(umbra=self,
                                          grid=gA,name="merged")
        my_layer.register_layers()
        self.log.info("Done with merging layers and registered result")

    def enable_tool(self):
        log.info("Enabled umbra mapTool")
        self.iface.mapCanvas().setMapTool(self.editor_tool)

    def run(self):
        """Run method that loads and starts the plugin"""
        self.log.info("** Call to run, which will activate")
        self.activate()

    def register_grid(self,ul):
        self.gridlayers.append(ul)
    def unregister_grid(self,ul):
        self.gridlayers.remove(ul)

    def grid_names(self):
        return [gl.name
                for gl in self.gridlayers]
    def generate_grid_name(self):
        existing=self.grid_names()
        for i in range(1000): # 1000 is arbitrary
            name="grid%d"%i
            if name not in existing:
                return name
        else:
            raise Exception("How can all of the names be taken")

    def name_to_grid(self,name):
        for gl in self.gridlayers:
            if gl.name==name:
                return gl
        return None

    def on_writeProject(self,doc):
        self.log.info("Got a writeProject(%s)"%(str(doc)))

        prj=QgsProject.instance()

        prj.writeEntry(scope,"GridCount",len(self.gridlayers))
        self.log.info("wrote GridCount=%d"%len(self.gridlayers))
        self.log.info("scope is %s"%scope)

        for i,gl in enumerate(self.gridlayers):
            if not gl.write_to_project(prj,scope,doc,"Grid%04d/"%i):
                msg="Umbra layer will not be saved in project -- insufficient information"
                self.iface.messageBar().pushMessage("Layer skipped",
                                                    msg,
                                                    level=Qgis.Info, duration=2)
                
    def on_readProject(self):
        self.log.info("Got a readProject signal")

        # not sure what the right level of cleanup is --
        # qgis takes care of cleaning up the legend.
        # at the very least, have to update our own state
        while self.gridlayers:
            # awkward, but avoids iterating on list while it's
            # being modified
            self.unregister_grid(self.gridlayers[0])

        self.log.info("readProject: starting with %d gridlayers"%len(self.gridlayers))

        prj=QgsProject.instance()

        grid_count,_ = prj.readNumEntry(scope,"GridCount",0)
        self.log.info("on_readProject: grid_count=%d"%grid_count)

        for i in range(grid_count):
            tag="Grid%04d/"%i
            self.log.info("on_readProject: reading from tag %s"%tag)

            # NB: gl may come back None
            gl=umbra_layer.UmbraLayer.load_from_project(self,prj,scope,tag)

            self.log.info("on_readProject: done with tag %s"%tag)
        if len(self.gridlayers):
            self.log.info("readProject: we have umbra layers, so activate plugin")
            self.activate()
        else:
            self.log.info("readProject: no umbra layers, so skip activate plugin")

