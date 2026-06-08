import typing

from py_trees import behaviour, common, composites


class ParallelPolicy(object):
    """Modified from py_trees.common.ParallelPolicy with SuccessOnK"""

    class Base(object):
        def __init__(self, synchronise: bool = False):
            self.synchronise = synchronise

    class SuccessOnAll(Base):
        def __init__(self, synchronise: bool = True):
            super().__init__(synchronise=synchronise)

    class SuccessOnOne(Base):
        def __init__(self) -> None:
            super().__init__(synchronise=False)

    class SuccessOnSelected(Base):
        def __init__(
            self,
            children: typing.List[typing.Any],
            synchronise: bool = True,
        ):
            super().__init__(synchronise=synchronise)
            self.children = children

    class SuccessOnK(Base):
        def __init__(
            self,
            k: int,
            synchronise: bool = True,
        ):
            super().__init__(synchronise=synchronise)
            self.k = k


class ParallelV2(composites.Composite):
    """modified from py_trees.composites.Parallel to include SuccessOnK policy"""

    def __init__(
        self,
        name: str,
        policy: ParallelPolicy.Base,
        children: typing.Optional[typing.Sequence[behaviour.Behaviour]] = None,
    ):
        super(ParallelV2, self).__init__(name, children)
        self.policy = policy

    def setup(self, **kwargs: typing.Any) -> None:
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        self.validate_policy_configuration()

    def tick(self) -> typing.Iterator[behaviour.Behaviour]:
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        self.validate_policy_configuration()

        # reset
        if self.status != common.Status.RUNNING:
            self.logger.debug("%s.tick(): re-initialising" % self.__class__.__name__)
            for child in self.children:
                # reset the children, this ensures old SUCCESS/FAILURE status flags
                # don't break the synchronisation logic below
                if child.status != common.Status.INVALID:
                    child.stop(common.Status.INVALID)
            self.current_child = None
            # subclass (user) handling
            self.initialise()

        # nothing to do
        if not self.children:
            self.current_child = None
            self.stop(common.Status.SUCCESS)
            yield self
            return

        # process them all first
        for child in self.children:
            if self.policy.synchronise and child.status == common.Status.SUCCESS:
                continue
            for node in child.tick():
                yield node

        # determine new status
        new_status = common.Status.RUNNING
        self.current_child = self.children[-1]
        try:
            failed_child = next(
                child
                for child in self.children
                if child.status == common.Status.FAILURE
            )
            self.current_child = failed_child
            new_status = common.Status.FAILURE
        except StopIteration:
            if isinstance(self.policy, ParallelPolicy.SuccessOnAll):
                if all(c.status == common.Status.SUCCESS for c in self.children):
                    new_status = common.Status.SUCCESS
                    self.current_child = self.children[-1]
            elif isinstance(self.policy, ParallelPolicy.SuccessOnOne):
                for child in reversed(self.children):
                    if child.status == common.Status.SUCCESS:
                        new_status = common.Status.SUCCESS
                        self.current_child = child
                        break
            elif isinstance(self.policy, ParallelPolicy.SuccessOnSelected):
                if all(
                    [c.status == common.Status.SUCCESS for c in self.policy.children]
                ):
                    new_status = common.Status.SUCCESS
                    self.current_child = self.policy.children[-1]
            elif isinstance(self.policy, ParallelPolicy.SuccessOnK):
                if (
                    sum(c.status == common.Status.SUCCESS for c in self.children)
                    >= self.policy.k
                ):
                    new_status = common.Status.SUCCESS
                    # set current child to the last child that succeeded
                    for child in reversed(self.children):
                        if child.status == common.Status.SUCCESS:
                            self.current_child = child
                            break
            else:
                raise RuntimeError(
                    "this parallel has been configured with an unrecognised policy [{}]".format(
                        type(self.policy)
                    )
                )
        # this parallel may have children that are still running
        # so if the parallel itself has reached a final status, then
        # these running children need to be terminated so they don't dangle
        if new_status != common.Status.RUNNING:
            self.stop(new_status)
        self.status = new_status
        yield self

    def stop(self, new_status: common.Status = common.Status.INVALID) -> None:
        """
        Ensure that any running children are stopped.

        Args:
            new_status : the composite is transitioning to this new status
        """
        self.logger.debug(
            f"{self.__class__.__name__}.stop()[{self.status}->{new_status}]"
        )

        # clean up dangling (running) children
        for child in self.children:
            if child.status == common.Status.RUNNING:
                # this unfortunately knocks out its running status for introspection
                # but logically is the correct thing to do, see #132.
                child.stop(common.Status.INVALID)
        composites.Composite.stop(self, new_status)

    def validate_policy_configuration(self) -> None:
        """
        Validate the currently stored policy.

        Policy configuration can be invalid if:
        * Policy is SuccessOnSelected and no behaviours have been specified
        * Policy is SuccessOnSelected and behaviours that are not children exist

        Raises:
            RuntimeError: if policy configuration was invalid
        """
        if isinstance(self.policy, ParallelPolicy.SuccessOnSelected):
            if not self.policy.children:
                error_message = (
                    "policy SuccessOnSelected requires a non-empty "
                    "selection of children [{}]".format(self.name)
                )
                self.logger.error(error_message)
                raise RuntimeError(error_message)
            missing_children_names = [
                child.name
                for child in self.policy.children
                if child not in self.children
            ]

            if missing_children_names:
                error_message = (
                    "policy SuccessOnSelected has selected behaviours that are "
                    "not children of this parallel {}[{}]"
                    "".format(missing_children_names, self.name)
                )
                self.logger.error(error_message)
                raise RuntimeError(error_message)
